from pathlib import Path

import pytest

from harness.multi_launch import Launch, launch_command
from harness.run import RunError


def L(n, same=False):
    return Launch(Path(f"/x/{n}.xclbin"), Path(f"/x/{n}.bin"), Path(f"/x/{n}.in"),
                  Path(f"/x/{n}.out"), 16, 32, same)


def test_each_launch_is_one_argument_in_order():
    cmd = launch_command(Path("/h"), [L("a"), L("b"), L("b2", same=True)])
    assert cmd == ["/h", "/x/a.xclbin,/x/a.bin,/x/a.in,/x/a.out,16,32,new",
                   "/x/b.xclbin,/x/b.bin,/x/b.in,/x/b.out,16,32,new",
                   "/x/b2.xclbin,/x/b2.bin,/x/b2.in,/x/b2.out,16,32,same"]


def test_the_first_launch_cannot_share_a_context():
    with pytest.raises(RunError):
        launch_command(Path("/h"), [L("a", same=True)])
    with pytest.raises(RunError):
        launch_command(Path("/h"), [])


def test_a_comma_in_a_path_is_refused():
    with pytest.raises(RunError):
        Launch(Path("/x/a,b"), Path("i"), Path("i"), Path("o"), 4, 4).arg()
