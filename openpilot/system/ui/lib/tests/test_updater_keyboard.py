import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pyray as rl
import raylib

from openpilot.common.basedir import BASEDIR
from openpilot.system.ui.lib.application import gui_app


class TestUpdaterKeyboard(unittest.TestCase):
  def test_selected_key_raylib_api(self):
    # The updater runs its bundled Python, not the working tree. Test both: a
    # source-only fix must not leave the pre-AGNOS-update password screen broken.
    keyboard_path = 'openpilot/system/ui/widgets/mici_keyboard.py'
    with zipfile.ZipFile(Path(BASEDIR) / 'openpilot/common/hardware/comma/updater') as bundle:
      bundled_source = bundle.read(keyboard_path)
    sources = {'source': (Path(BASEDIR) / keyboard_path).read_bytes(), 'bundle': bundled_source}
    signatures = {'installed': raylib.ffi.typeof(raylib.DrawCircleGradient),
                  'raylib5': raylib.ffi.typeof('void(*)(int, int, float, struct Color, struct Color)')}

    for name, source in sources.items():
      for version, signature in signatures.items():
        with self.subTest(source=name, raylib=version):
          calls = []
          # Enforce the C ABI without an OpenGL context. A plain Mock silently
          # accepts the wrong argument count. Also retain old-OS compatibility.
          draw_gradient = raylib.ffi.callback(signature, lambda *args, _calls=calls: _calls.append(len(args)))
          texture = rl.Texture(0, 520, 170, 1, 0)
          with patch.object(raylib, 'DrawCircleGradient', draw_gradient), \
               patch.object(gui_app, 'font'), patch.object(gui_app, 'texture', return_value=texture), \
               patch.object(rl, 'draw_circle_gradient', draw_gradient):
            module = types.ModuleType('updater_keyboard')
            exec(compile(source, f'{name}/mici_keyboard.py', 'exec'), module.__dict__)
            keyboard = module.MiciKeyboard()
            keyboard.set_rect(rl.Rectangle(0, 0, 540, 300))
            keyboard._closest_key = (keyboard._lower_keys[0][0], 0)
            keyboard._lay_out_keys(0, 130, keyboard._current_keys)
          self.assertEqual(calls, [len(signature.args)])
