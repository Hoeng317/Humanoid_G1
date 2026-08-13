#include "humanoid_g1/controller.hpp"

#include <ATen/Parallel.h>
#include <torch/script.h>
#include <yaml-cpp/yaml.h>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using humanoid_g1::Contract;
using humanoid_g1::ControllerState;
using humanoid_g1::NeutralState;
using humanoid_g1::ObservationHistory;
using humanoid_g1::StateMachine;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

namespace {

std::atomic<bool> running{true};
void signal_handler(int) { running = false; }

struct Options {
  std::filesystem::path policy;
  std::filesystem::path golden_csv;
  std::filesystem::path hardware_profile;
  std::string interface{"lo"};
  int domain_id{1};
  bool sim{false};
  bool real{false};
  bool acknowledge{false};
  bool stand{false};
  bool run_policy{false};
  bool dry_run{false};
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string value = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value after " + value);
      return argv[i];
    };
    if (value == "--policy") options.policy = next();
    else if (value == "--golden-csv") options.golden_csv = next();
    else if (value == "--hardware-profile") options.hardware_profile = next();
    else if (value == "--interface") options.interface = next();
    else if (value == "--domain-id") options.domain_id = std::stoi(next());
    else if (value == "--sim") options.sim = true;
    else if (value == "--real") options.real = true;
    else if (value == "--acknowledge-hardware-risk") options.acknowledge = true;
    else if (value == "--stand") options.stand = true;
    else if (value == "--run") options.run_policy = true;
    else if (value == "--dry-run") options.dry_run = true;
    else if (value == "--help") {
      std::cout << "g1_controller --policy FILE [--golden-csv FILE --dry-run]\n"
                   "  sim:  --sim --interface lo --domain-id 1 [--stand|--run]\n"
                   "  real: --real --acknowledge-hardware-risk --interface IFACE "
                   "--domain-id 0 --hardware-profile FILE [--stand|--run]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + value);
    }
  }
  if (options.policy.empty()) throw std::runtime_error("--policy is required");
  return options;
}

void validate_mode(const Options& options) {
  if (!std::filesystem::is_regular_file(options.policy)) throw std::runtime_error("policy not found");
  if (options.sim && options.real) throw std::runtime_error("--sim and --real are mutually exclusive");
  if (options.sim && (options.interface != "lo" || options.domain_id != 1)) {
    throw std::runtime_error("simulation transport is restricted to interface lo and DDS domain 1");
  }
  if (options.real) {
    if (!options.acknowledge) throw std::runtime_error("real mode requires risk acknowledgement");
    if (options.interface.empty() || options.interface == "lo" || options.domain_id != 0) {
      throw std::runtime_error("real mode requires non-loopback interface and DDS domain 0");
    }
    if (!std::filesystem::exists("/sys/class/net/" + options.interface)) {
      throw std::runtime_error("real network interface does not exist");
    }
    if (options.hardware_profile.empty()) throw std::runtime_error("real mode requires hardware profile");
    const auto profile = YAML::LoadFile(options.hardware_profile.string());
    if (!profile["hardware"]["verified"].as<bool>() ||
        !profile["hardware"]["motor_index_verified"].as<bool>()) {
      throw std::runtime_error("hardware profile or motor index is not verified");
    }
  }
  if (!options.sim && !options.real && !options.dry_run) {
    throw std::runtime_error("transport requires an explicit --sim or --real mode");
  }
}

float run_golden(torch::jit::script::Module& policy, const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("golden CSV not found");
  float max_error = 0.0F;
  std::size_t samples = 0;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::vector<float> values;
    std::string token;
    while (std::getline(stream, token, ',')) values.push_back(std::stof(token));
    if (values.size() != humanoid_g1::kObservationDim + humanoid_g1::kNumJoints) {
      throw std::runtime_error("golden CSV row dimension mismatch");
    }
    auto tensor = torch::from_blob(values.data(), {1, 480}, torch::kFloat32).clone();
    const auto action = policy.forward({tensor}).toTensor().contiguous();
    const float* result = action.data_ptr<float>();
    for (std::size_t i = 0; i < 29; ++i) {
      max_error = std::max(max_error, std::abs(result[i] - values[480 + i]));
    }
    ++samples;
  }
  std::cout << "C++ golden vectors: samples=" << samples << ", max_abs_error=" << max_error << '\n';
  return max_error;
}

class TransportController {
 public:
  TransportController(const Options& options, torch::jit::script::Module policy)
      : options_(options), policy_(std::move(policy)) {}

  void run() {
    unitree::robot::ChannelFactory::Instance()->Init(options_.domain_id, options_.interface);
    subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<LowState_>>("rt/lowstate");
    subscriber_->InitChannel([this](const void* message) { on_state(message); }, 1);
    std::cout << "State: DISCONNECTED; waiting for verified G1 LowState on rt/lowstate\n";
    {
      std::unique_lock lock(mutex_);
      if (!condition_.wait_for(lock, std::chrono::seconds(5), [this] { return have_state_; })) {
        throw std::runtime_error("LowState timeout; LowCmd publisher was not created");
      }
    }
    machine_.transition(ControllerState::IDLE);
    std::cout << "State: IDLE; valid G1 LowState received\n";
    if (!options_.stand && !options_.run_policy) {
      std::cout << "No --stand/--run operator command; remaining IDLE without LowCmd publisher\n";
      return;
    }
    // The publisher is constructed only after transport/state/mode checks and explicit stand/run input.
    publisher_ = std::make_shared<unitree::robot::ChannelPublisher<LowCmd_>>("rt/lowcmd");
    publisher_->InitChannel();
    machine_.transition(ControllerState::DAMPING);
    machine_.transition(ControllerState::MOVE_TO_DEFAULT);
    const auto transition_start = std::chrono::steady_clock::now();
    try {
      while (running) {
        const auto loop_start = std::chrono::steady_clock::now();
        NeutralState state;
        uint8_t mode_machine = 0;
        {
          std::lock_guard lock(mutex_);
          state = state_;
          mode_machine = mode_machine_;
        }
        validate_state(state, loop_start);
        const double stand_time = std::chrono::duration<double>(loop_start - transition_start).count();
        if (machine_.state() == ControllerState::MOVE_TO_DEFAULT) {
          send_default(state, mode_machine, std::clamp(stand_time / 3.0, 0.0, 1.0));
          if (stand_time >= 3.0) {
            machine_.transition(ControllerState::STAND_READY);
            std::cout << "State: STAND_READY\n";
            if (options_.run_policy) {
              machine_.transition(ControllerState::POLICY_READY);
              machine_.transition(ControllerState::POLICY_RUNNING, true);
              std::cout << "State: POLICY_RUNNING (explicit --run)\n";
            }
          }
        } else if (machine_.state() == ControllerState::STAND_READY) {
          send_default(state, mode_machine, 1.0);
        } else if (machine_.state() == ControllerState::POLICY_RUNNING) {
          send_policy(state, mode_machine);
        }
        const double work_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
        if (work_time > 0.03) throw std::runtime_error("control loop overrun");
        std::this_thread::sleep_until(loop_start + std::chrono::milliseconds(20));
      }
    } catch (const std::exception& error) {
      machine_.fault(error.what());
      std::cerr << "State: FAULT (" << error.what() << "); sending damping command\n";
      NeutralState state;
      uint8_t mode_machine = 0;
      {
        std::lock_guard lock(mutex_);
        state = state_;
        mode_machine = mode_machine_;
      }
      send_damping(state, mode_machine);
      throw;
    }
    if (publisher_) {
      machine_.transition(ControllerState::STOPPING);
      send_damping(state_, mode_machine_);
      std::cout << "State: STOPPING; final damping command sent\n";
    }
  }

 private:
  void validate_state(
      const NeutralState& state, std::chrono::steady_clock::time_point now) {
    const double age = std::chrono::duration<double>(now - state.received_at).count();
    if (age > 0.1) throw std::runtime_error("LowState stale or network disconnected");
    // Unitree's reference controller uses L2+B as the operator stop chord.
    constexpr uint16_t kL2 = 1U << 5U;
    constexpr uint16_t kB = 1U << 9U;
    if ((state.remote_buttons & kL2) && (state.remote_buttons & kB)) {
      throw std::runtime_error("remote emergency stop (L2+B)");
    }
    for (std::size_t policy = 0; policy < 29; ++policy) {
      const std::size_t sdk = Contract::policy_to_sdk[policy];
      const float q = state.q_sdk[sdk];
      const float dq = state.dq_sdk[sdk];
      if (!std::isfinite(q) || !std::isfinite(dq)) {
        throw std::runtime_error("non-finite joint state");
      }
      if (q < Contract::lower_policy[policy] || q > Contract::upper_policy[policy]) {
        throw std::runtime_error("joint position limit exceeded");
      }
      if (std::abs(dq) > Contract::velocity_policy[policy]) {
        throw std::runtime_error("joint velocity limit exceeded");
      }
    }
    const auto& q = state.quaternion_wxyz;
    const float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!std::isfinite(norm) || norm < 0.8F || norm > 1.2F) {
      throw std::runtime_error("invalid IMU quaternion");
    }
    const float w = q[0] / norm;
    const float x = q[1] / norm;
    const float y = q[2] / norm;
    const float z = q[3] / norm;
    const float roll = std::atan2(2.0F * (w * x + y * z), 1.0F - 2.0F * (x * x + y * y));
    const float pitch_argument = std::clamp(2.0F * (w * y - z * x), -1.0F, 1.0F);
    const float pitch = std::asin(pitch_argument);
    if (std::abs(roll) > 0.7F || std::abs(pitch) > 0.7F) {
      throw std::runtime_error("body orientation limit exceeded");
    }
  }

  void on_state(const void* message) {
    const auto& raw = *static_cast<const LowState_*>(message);
    if (raw.crc() != humanoid_g1::crc32_core(
                         const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(&raw)),
                         (sizeof(LowState_) >> 2U) - 1U)) return;
    NeutralState next;
    for (std::size_t i = 0; i < 29; ++i) {
      next.q_sdk[i] = raw.motor_state()[i].q();
      next.dq_sdk[i] = raw.motor_state()[i].dq();
    }
    next.quaternion_wxyz = raw.imu_state().quaternion();
    next.gyro = raw.imu_state().gyroscope();
    const auto& remote = raw.wireless_remote();
    next.remote_buttons = static_cast<uint16_t>(remote[2]) |
                          (static_cast<uint16_t>(remote[3]) << 8U);
    next.received_at = std::chrono::steady_clock::now();
    {
      std::lock_guard lock(mutex_);
      state_ = next;
      mode_machine_ = raw.mode_machine();
      have_state_ = true;
    }
    condition_.notify_one();
  }

  LowCmd_ base_command(uint8_t mode_machine) {
    LowCmd_ command;
    command.mode_pr() = 0;
    command.mode_machine() = mode_machine;
    return command;
  }

  void finish_and_write(LowCmd_& command) {
    command.crc() = humanoid_g1::crc32_core(
        reinterpret_cast<uint32_t*>(&command), (sizeof(LowCmd_) >> 2U) - 1U);
    publisher_->Write(command);
  }

  void send_damping(const NeutralState& state, uint8_t mode_machine) {
    auto command = base_command(mode_machine);
    for (std::size_t sdk = 0; sdk < 29; ++sdk) {
      command.motor_cmd()[sdk].mode() = 1;
      command.motor_cmd()[sdk].q() = state.q_sdk[sdk];
      command.motor_cmd()[sdk].dq() = 0;
      command.motor_cmd()[sdk].kp() = 0;
      command.motor_cmd()[sdk].kd() = 3;
      command.motor_cmd()[sdk].tau() = 0;
    }
    finish_and_write(command);
  }

  void send_default(const NeutralState& state, uint8_t mode_machine, double ratio) {
    auto command = base_command(mode_machine);
    for (std::size_t policy = 0; policy < 29; ++policy) {
      const std::size_t sdk = Contract::policy_to_sdk[policy];
      command.motor_cmd()[sdk].mode() = 1;
      command.motor_cmd()[sdk].q() =
          static_cast<float>((1.0 - ratio) * state.q_sdk[sdk] + ratio * Contract::default_policy[policy]);
      command.motor_cmd()[sdk].dq() = 0;
      command.motor_cmd()[sdk].kp() = Contract::kp_policy[policy];
      command.motor_cmd()[sdk].kd() = Contract::kd_policy[policy];
      command.motor_cmd()[sdk].tau() = 0;
    }
    finish_and_write(command);
  }

  void send_policy(const NeutralState& state, uint8_t mode_machine) {
    const auto observation = history_.update(state, {0, 0, 0}, previous_action_);
    auto tensor = torch::from_blob(
                      const_cast<float*>(observation.data()), {1, 480}, torch::kFloat32)
                      .clone();
    const auto start = std::chrono::steady_clock::now();
    const auto output = policy_.forward({tensor}).toTensor().contiguous();
    const double inference = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
    if (inference > 0.01) {
      throw std::runtime_error(
          "policy inference timeout: " + std::to_string(inference * 1000.0) + " ms");
    }
    std::array<float, 29> action{};
    std::copy_n(output.data_ptr<float>(), 29, action.begin());
    const auto targets = humanoid_g1::bounded_targets(action, previous_action_);
    auto command = base_command(mode_machine);
    for (std::size_t policy = 0; policy < 29; ++policy) {
      const std::size_t sdk = Contract::policy_to_sdk[policy];
      command.motor_cmd()[sdk].mode() = 1;
      command.motor_cmd()[sdk].q() = targets[sdk];
      command.motor_cmd()[sdk].dq() = 0;
      command.motor_cmd()[sdk].kp() = Contract::kp_policy[policy];
      command.motor_cmd()[sdk].kd() = Contract::kd_policy[policy];
      command.motor_cmd()[sdk].tau() = 0;
    }
    finish_and_write(command);
  }

  Options options_;
  torch::jit::script::Module policy_;
  StateMachine machine_;
  ObservationHistory history_;
  std::array<float, 29> previous_action_{};
  std::mutex mutex_;
  std::condition_variable condition_;
  NeutralState state_;
  uint8_t mode_machine_{0};
  bool have_state_{false};
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState_>> subscriber_;
  std::shared_ptr<unitree::robot::ChannelPublisher<LowCmd_>> publisher_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    const Options options = parse_options(argc, argv);
    validate_mode(options);
    auto policy = torch::jit::load(options.policy.string(), torch::kCPU);
    policy.eval();
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    const auto warmup = torch::zeros({1, 480});
    for (int i = 0; i < 20; ++i) policy.forward({warmup});
    if (!options.golden_csv.empty()) {
      const float error = run_golden(policy, options.golden_csv);
      if (error > 1e-5F) throw std::runtime_error("C++ golden-vector tolerance exceeded");
    }
    if (options.dry_run) {
      std::cout << "Dry-run passed; no DDS factory or publisher was created\n";
      return 0;
    }
    TransportController controller(options, std::move(policy));
    controller.run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "g1_controller: " << error.what() << '\n';
    return 1;
  }
}
