from humanoid_g1.assets.joint_contract import load_joint_contract


def test_contract_has_exact_29_dof_and_permutations():
    contract = load_joint_contract()
    assert contract.action_dim == 29
    assert sorted(contract.policy_to_sdk) == list(range(29))
    assert sorted(contract.sdk_to_policy) == list(range(29))
    assert contract.policy_dt == 0.02


def test_mapping_round_trip():
    contract = load_joint_contract()
    source = [float(index) for index in range(29)]
    assert contract.reorder_sdk_to_policy(contract.reorder_policy_to_sdk(source)) == source


def test_official_assets_match_contract():
    contract = load_joint_contract()
    contract.validate_urdf()
    contract.validate_mujoco()

