#include "humanoid_g1/controller.hpp"

#include <algorithm>
#include <cmath>

namespace humanoid_g1 {

const char* state_name(ControllerState state) {
  switch (state) {
    case ControllerState::DISCONNECTED: return "DISCONNECTED";
    case ControllerState::IDLE: return "IDLE";
    case ControllerState::DAMPING: return "DAMPING";
    case ControllerState::ZERO_TORQUE: return "ZERO_TORQUE";
    case ControllerState::MOVE_TO_DEFAULT: return "MOVE_TO_DEFAULT";
    case ControllerState::STAND_READY: return "STAND_READY";
    case ControllerState::POLICY_READY: return "POLICY_READY";
    case ControllerState::POLICY_RUNNING: return "POLICY_RUNNING";
    case ControllerState::STOPPING: return "STOPPING";
    case ControllerState::FAULT: return "FAULT";
  }
  return "UNKNOWN";
}

void StateMachine::transition(ControllerState target, bool explicit_operator) {
  bool allowed = false;
  switch (state_) {
    case ControllerState::DISCONNECTED:
      allowed = target == ControllerState::IDLE || target == ControllerState::FAULT;
      break;
    case ControllerState::IDLE:
      allowed = target == ControllerState::DAMPING || target == ControllerState::ZERO_TORQUE ||
                target == ControllerState::FAULT;
      break;
    case ControllerState::DAMPING:
      allowed = target == ControllerState::MOVE_TO_DEFAULT || target == ControllerState::STOPPING ||
                target == ControllerState::FAULT;
      break;
    case ControllerState::ZERO_TORQUE:
      allowed = target == ControllerState::DAMPING || target == ControllerState::STOPPING ||
                target == ControllerState::FAULT;
      break;
    case ControllerState::MOVE_TO_DEFAULT:
      allowed = target == ControllerState::STAND_READY || target == ControllerState::STOPPING ||
                target == ControllerState::FAULT;
      break;
    case ControllerState::STAND_READY:
      allowed = target == ControllerState::POLICY_READY || target == ControllerState::STOPPING ||
                target == ControllerState::FAULT;
      break;
    case ControllerState::POLICY_READY:
      allowed = target == ControllerState::POLICY_RUNNING || target == ControllerState::STOPPING ||
                target == ControllerState::FAULT;
      break;
    case ControllerState::POLICY_RUNNING:
      allowed = target == ControllerState::STOPPING || target == ControllerState::FAULT;
      break;
    case ControllerState::STOPPING:
      allowed = target == ControllerState::DAMPING || target == ControllerState::IDLE ||
                target == ControllerState::FAULT;
      break;
    case ControllerState::FAULT:
      allowed = target == ControllerState::DAMPING || target == ControllerState::IDLE;
      break;
  }
  if (!allowed) throw std::runtime_error("invalid controller state transition");
  if (target == ControllerState::POLICY_RUNNING && !explicit_operator) {
    throw std::runtime_error("POLICY_RUNNING requires explicit operator input");
  }
  state_ = target;
  reason_.clear();
}

void StateMachine::fault(const std::string& reason) {
  state_ = ControllerState::FAULT;
  reason_ = reason;
}

std::array<float, kObservationDim> ObservationHistory::update(
    const NeutralState& state,
    const std::array<float, 3>& command,
    const std::array<float, kNumJoints>& last_action) {
  const auto& quat = state.quaternion_wxyz;
  const float norm = std::sqrt(
      quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  if (!std::isfinite(norm) || norm < 1e-8F) throw std::runtime_error("invalid IMU quaternion");
  const float w = quat[0] / norm;
  const float x = quat[1] / norm;
  const float y = quat[2] / norm;
  const float z = quat[3] / norm;
  const std::array<float, 3> gravity{
      2.0F * (x * z - w * y),
      2.0F * (y * z + w * x),
      -(1.0F - 2.0F * (x * x + y * y))};

  std::array<float, kFrameDim> frame{};
  std::size_t cursor = 0;
  for (float value : state.gyro) frame[cursor++] = 0.2F * value;
  for (float value : gravity) frame[cursor++] = value;
  for (float value : command) frame[cursor++] = value;
  for (std::size_t policy = 0; policy < kNumJoints; ++policy) {
    frame[cursor++] = state.q_sdk[Contract::policy_to_sdk[policy]] - Contract::default_policy[policy];
  }
  for (std::size_t policy = 0; policy < kNumJoints; ++policy) {
    frame[cursor++] = 0.05F * state.dq_sdk[Contract::policy_to_sdk[policy]];
  }
  for (float value : last_action) frame[cursor++] = value;
  if (cursor != kFrameDim) throw std::runtime_error("observation frame contract mismatch");
  if (frames_.empty()) {
    for (int i = 0; i < 5; ++i) frames_.push_back(frame);
  } else {
    frames_.pop_front();
    frames_.push_back(frame);
  }
  std::array<float, kObservationDim> output{};
  cursor = 0;
  for (const auto& stored : frames_) {
    for (float value : stored) output[cursor++] = value;
  }
  return output;
}

std::array<float, kNumJoints> bounded_targets(
    const std::array<float, kNumJoints>& action,
    std::array<float, kNumJoints>& previous_action) {
  std::array<float, kNumJoints> target_sdk{};
  for (std::size_t policy = 0; policy < kNumJoints; ++policy) {
    if (!std::isfinite(action[policy])) throw std::runtime_error("non-finite policy action");
    const float clipped = std::clamp(action[policy], -1.0F, 1.0F);
    const float rate_limited = std::clamp(
        clipped, previous_action[policy] - 0.15F, previous_action[policy] + 0.15F);
    previous_action[policy] = rate_limited;
    const float target = std::clamp(
        Contract::default_policy[policy] + 0.25F * rate_limited,
        Contract::lower_policy[policy] + 0.05F,
        Contract::upper_policy[policy] - 0.05F);
    target_sdk[Contract::policy_to_sdk[policy]] = target;
  }
  return target_sdk;
}

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xFFFFFFFF;
  constexpr uint32_t polynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t xbit = 1U << 31;
    const uint32_t data = ptr[i];
    for (uint32_t bit = 0; bit < 32; ++bit) {
      crc = (crc & 0x80000000U) ? (crc << 1U) ^ polynomial : crc << 1U;
      if (data & xbit) crc ^= polynomial;
      xbit >>= 1U;
    }
  }
  return crc;
}

}  // namespace humanoid_g1

