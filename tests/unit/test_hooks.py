from humanoid_g1.interfaces.hooks import (
    ConstantCommandProvider,
    ExternalPolicyCommandProvider,
    IdentitySafetyFilter,
    ScriptedCommandProvider,
)


def test_command_providers():
    assert tuple(ConstantCommandProvider((1, 2, 3)).command(10.0)) == (1.0, 2.0, 3.0)
    scripted = ScriptedCommandProvider([(0.0, (0, 0, 0)), (2.0, (1, 0, 0))])
    assert tuple(scripted.command(1.0)) == (0.0, 0.0, 0.0)
    assert tuple(scripted.command(3.0)) == (1.0, 0.0, 0.0)
    external = ExternalPolicyCommandProvider()
    external.update((0.2, -0.1, 0.0), 1.0)
    assert tuple(external.command(2.0)) == (0.2, -0.1, 0.0)


def test_identity_safety_filter_contract():
    command, info = IdentitySafetyFilter().filter(
        observation={}, nominal_command=(0.1, 0.0, 0.0), timestamp=1.0
    )
    assert command == [0.1, 0.0, 0.0]
    assert info == {"modified": False}
