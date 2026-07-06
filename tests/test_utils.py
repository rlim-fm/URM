import pytest

from utils import load_model_class, get_model_source_path
from tests.conftest import requires_flash_attn, requires_numba


@requires_numba
def test_load_evaluator_class_with_at_identifier():
    cls = load_model_class("arc@ARC", prefix="evaluators.")
    assert cls.__name__ == "ARC"


def test_load_model_class_bad_identifier_raises():
    with pytest.raises(ModuleNotFoundError):
        load_model_class("no_such_module@Whatever")


def test_get_model_source_path_with_at_identifier():
    path = get_model_source_path("losses@ACTLossHead", prefix="models.")
    assert path.endswith("losses.py")


def test_bare_module_infers_class_by_convention():
    # models/losses.py has no bare "losses" class, but has ACTLossHead;
    # verify the inference candidates logic raises a descriptive error
    # rather than silently picking the wrong thing.
    with pytest.raises(ValueError, match="Could not infer class"):
        load_model_class("losses", prefix="models.")


@requires_flash_attn
def test_load_model_class_urm_with_at_identifier():
    cls = load_model_class("urm.urm@URM")
    assert cls.__name__ == "URM"
