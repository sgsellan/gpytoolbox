from gpytoolbox_bindings import viewer
import gpytoolbox as gpy
import numpy as np

name = "airplane"

V, F, UV_py, Ft_py, N_py, Fn_py = \
    gpy.read_mesh("test/unit_tests_data/" + name,
                  return_UV=True, return_N=True, reader='Python')
def callback_pre_draw():
    print("callback pre draw")
    return False

def key_pressed_callback(button, modifier):
    print("callback_pre_draw")
    print("key pressed  : ", chr(button))
    print("key pressed  : ", modifier)
    return False

def mouse_pressed_callback(button, modifier):
    print("mouse_pressed_callback")
    print("mouse pressed  : ", button)
    print("mouse pressed  : ", modifier)
    return False

def callback_mouse_scroll(dy):
    print("zoom : ", dy)
    return False

def callback_post_draw():
    print("callback_post_draw")
    return False

v = viewer()
v.background_color(np.array([1,1,1, 1.0]))
v.set_mesh(V, F)
v.show_lines(False)

v.callback_pre_draw(callback_pre_draw)
v.callback_post_draw(callback_post_draw)
v.callback_mouse_down(mouse_pressed_callback)
v.callback_key_pressed(key_pressed_callback)
v.callback_mouse_scroll(callback_mouse_scroll)
v.launch()




