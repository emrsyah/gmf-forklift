from src.capture import apply_v4l2_controls, v4l2_ctl_args


def test_args_keep_config_order():
    args = v4l2_ctl_args({"focus_automatic_continuous": 0, "focus_absolute": 0, "power_line_frequency": 1})
    assert args == [["--set-ctrl", "focus_automatic_continuous=0"], ["--set-ctrl", "focus_absolute=0"],
                    ["--set-ctrl", "power_line_frequency=1"]]


def test_no_controls_is_noop():
    assert v4l2_ctl_args(None) == []
    apply_v4l2_controls(0, None)  # tidak memanggil apa pun
