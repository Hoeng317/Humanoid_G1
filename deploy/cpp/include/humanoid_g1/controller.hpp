#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>

namespace humanoid_g1 {

constexpr std::size_t kNumJoints = 29;
constexpr std::size_t kFrameDim = 96;
constexpr std::size_t kObservationDim = 480;

enum class ControllerState {
  DISCONNECTED,
  IDLE,
  DAMPING,
  ZERO_TORQUE,
  MOVE_TO_DEFAULT,
  STAND_READY,
  POLICY_READY,
  POLICY_RUNNING,
  STOPPING,
  FAULT,
};

struct Contract {
  // Policy index -> SDK/hardware motor index.
  static constexpr std::array<int, kNumJoints> policy_to_sdk{
      0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
      16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};
  static constexpr std::array<float, kNumJoints> default_policy{
      -0.1F, -0.1F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
      0.3F, 0.3F, 0.3F, 0.3F, -0.2F, -0.2F, 0.25F, -0.25F, 0.0F,
      0.0F, 0.0F, 0.0F, 0.97F, 0.97F, 0.15F, -0.15F, 0.0F, 0.0F,
      0.0F, 0.0F};
  static constexpr std::array<float, kNumJoints> lower_policy{
      -2.5307F, -2.5307F, -2.618F, -0.5236F, -2.9671F, -0.52F,
      -2.7576F, -2.7576F, -0.52F, -0.087267F, -0.087267F, -3.0892F,
      -3.0892F, -0.87267F, -0.87267F, -1.5882F, -2.2515F, -0.2618F,
      -0.2618F, -2.618F, -2.618F, -1.0472F, -1.0472F, -1.972222054F,
      -1.972222054F, -1.614429558F, -1.614429558F, -1.614429558F,
      -1.614429558F};
  static constexpr std::array<float, kNumJoints> upper_policy{
      2.8798F, 2.8798F, 2.618F, 2.9671F, 0.5236F, 0.52F, 2.7576F,
      2.7576F, 0.52F, 2.8798F, 2.8798F, 2.6704F, 2.6704F, 0.5236F,
      0.5236F, 2.2515F, 1.5882F, 0.2618F, 0.2618F, 2.618F, 2.618F,
      2.0944F, 2.0944F, 1.972222054F, 1.972222054F, 1.614429558F,
      1.614429558F, 1.614429558F, 1.614429558F};
  static constexpr std::array<float, kNumJoints> kp_policy{
      100, 100, 200, 100, 100, 40, 100, 100, 40, 150, 150, 40, 40,
      40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40};
  static constexpr std::array<float, kNumJoints> kd_policy{
      2, 2, 5, 2, 2, 5, 2, 2, 5, 4, 4, 1, 1, 2, 2,
      1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  static constexpr std::array<float, kNumJoints> velocity_policy{
      32, 32, 32, 20, 20, 30, 32, 32, 30, 20, 20, 37, 37, 30, 30,
      37, 37, 30, 30, 37, 37, 37, 37, 37, 37, 22, 22, 22, 22};
};

class StateMachine {
 public:
  ControllerState state() const { return state_; }
  void transition(ControllerState target, bool explicit_operator = false);
  void fault(const std::string& reason);
  const std::string& reason() const { return reason_; }

 private:
  ControllerState state_{ControllerState::DISCONNECTED};
  std::string reason_;
};

struct NeutralState {
  std::array<float, kNumJoints> q_sdk{};
  std::array<float, kNumJoints> dq_sdk{};
  std::array<float, 4> quaternion_wxyz{1, 0, 0, 0};
  std::array<float, 3> gyro{};
  uint16_t remote_buttons{0};
  std::chrono::steady_clock::time_point received_at{};
};

class ObservationHistory {
 public:
  std::array<float, kObservationDim> update(
      const NeutralState& state,
      const std::array<float, 3>& command,
      const std::array<float, kNumJoints>& last_action);
  void reset() { frames_.clear(); }

 private:
  std::deque<std::array<float, kFrameDim>> frames_;
};

std::array<float, kNumJoints> bounded_targets(
    const std::array<float, kNumJoints>& action,
    std::array<float, kNumJoints>& previous_action);

uint32_t crc32_core(uint32_t* ptr, uint32_t len);
const char* state_name(ControllerState state);

}  // namespace humanoid_g1
