"""The ball's CONTACT model must stay identical to the shared asset (P1-T7).

``ball.xml`` is the asset every BallKick-trained policy has seen, and the
soccer env deliberately adds ONLY rolling resistance (a freejoint
``frictionloss``). If someone "improves realism" by editing ``geom.condim`` /
``geom.friction`` in the spec builders, ball-ground and ball-foot contact stop
matching the trained asset — and the difference is invisible in the viewer.
"""

import pathlib

import mujoco

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_XML
from mjlab_microduck.soccer import pitch as P


def _ball_geom(model: mujoco.MjModel) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    assert i >= 0
    return i


def test_ball_xml_itself_is_unmodified():
    """The calibration line: the file on disk still says condim=3."""
    text = pathlib.Path(MICRODUCK_BALL_XML).read_text()
    assert 'condim' not in text, "ball.xml must not pin condim explicitly"
    assert 'friction="0.5 0.005 0.0001"' in text
    assert 'frictionloss' not in text, "rolling resistance belongs in the spec, not the file"


def test_compiled_ball_keeps_condim_and_friction():
    model = P.ball_spec().compile()
    g = _ball_geom(model)
    assert model.geom_condim[g] == 3
    assert tuple(model.geom_friction[g]) == (0.5, 0.005, 0.0001)


def test_compiled_ball_matches_a_plain_load_of_the_asset():
    """Bit-identical contact parameters to an untouched ``MjSpec.from_file``."""
    plain = mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML)).compile()
    ours = P.ball_spec().compile()
    gp, go = _ball_geom(plain), _ball_geom(ours)
    assert plain.geom_condim[gp] == ours.geom_condim[go]
    assert (plain.geom_friction[gp] == ours.geom_friction[go]).all()
    assert (plain.geom_size[gp] == ours.geom_size[go]).all()
    assert (plain.geom_solref[gp] == ours.geom_solref[go]).all()


def test_the_only_difference_is_rolling_resistance():
    plain = mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML)).compile()
    ours = P.ball_spec().compile()
    # Untouched asset: no joint friction at all.
    assert (plain.dof_frictionloss == 0.0).all()
    assert (ours.dof_frictionloss == P.BALL_ROLL_LOSS).all()
    # Everything else that shapes the dynamics is unchanged.
    assert plain.nv == ours.nv == 6
    ball = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_o = mujoco.mj_name2id(ours, mujoco.mjtObj.mjOBJ_BODY, "ball")
    assert plain.body_mass[ball] == ours.body_mass[ball_o] == 0.015
    assert (plain.body_inertia[ball] == ours.body_inertia[ball_o]).all()


def test_pitch_world_ball_is_unchanged_too():
    """Attaching the ball into the pitch spec (prefixed / framed) changes nothing."""
    spec = P.pitch_spec()
    spec.attach(P.ball_spec(), prefix="", frame=spec.worldbody.add_frame(pos=[0, 0, 0]))
    model = spec.compile()
    assert model.geom_condim[_ball_geom(model)] == 3
    assert (model.dof_frictionloss == P.BALL_ROLL_LOSS).all()
