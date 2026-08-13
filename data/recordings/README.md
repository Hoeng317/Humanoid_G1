# Robot/simulator recordings

가공 전 LowState 또는 Isaac rollout을 보관합니다. timestamp, SDK 순서의 q/dq,
quaternion(wxyz), gyro, command, 이전 action을 함께 기록해야 나중에 480차원 actor
observation을 재구성할 수 있습니다.
