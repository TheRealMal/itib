import dearpygui.dearpygui as dpg
from RNN import SAVED_IMAGES, RecurrentNNetwork

NET = RecurrentNNetwork(SAVED_IMAGES)
CHECK_IMAGE = [0 for _ in range(15)]

dpg.create_context()

#
# Темы для сохраненных образов
#
with dpg.theme() as white_cell:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255])

with dpg.theme() as black_cell:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Text, [0, 0, 0])

with dpg.theme() as green_window:
    with dpg.theme_component(dpg.mvWindowAppItem):
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg, [23,162,95,255])
        dpg.add_theme_color(dpg.mvThemeCol_Border, [23,162,95,255])
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, [23,162,95,255])

#
# Вывод окон с сохраненными образами
#
for i in range(len(SAVED_IMAGES)):
    with dpg.window(
        label="Image #{}".format(i + 1),
        pos=(i*100, 0),
        no_resize=True,
        no_move=True,
        no_close=True,
        no_collapse=True,
    ) as w:
        dpg.bind_item_theme(w, green_window)
        with dpg.table(header_row=False):
            for _ in range(3):
                dpg.add_table_column()
            
            for row in range(5):
                with dpg.table_row():
                    for col in range(3):
                        cell = dpg.add_text("X")
                        if SAVED_IMAGES[i][col * 5 + row] > 0:
                            dpg.bind_item_theme(cell, white_cell)
                        else:
                            dpg.bind_item_theme(cell, black_cell)

#
# Callback функции для проверки образа
#
def check_table_callback(sender, app_data, user_data) -> None:
    if app_data:
        CHECK_IMAGE[user_data] = 1
    else:
        CHECK_IMAGE[user_data] = -1

def check_result(sendes, app_data, user_data) -> None:
    if user_data:
        res = NET.recover_image(CHECK_IMAGE, "sync")
    else:
        res = NET.recover_image(CHECK_IMAGE, "async")
    dpg.configure_item("result", default_value=res)

#
# Темы для окна с проверкой образа
#
with dpg.theme() as check_window:
    with dpg.theme_component(dpg.mvWindowAppItem):
        dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255])
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, [23,162,95,255])
    
with dpg.theme() as check_cell:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255])

with dpg.theme() as check_buttons:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Button, [23,162,95,255])

with dpg.window(
    label="Check image",
    pos=(0, 150),
    no_resize=True,
    no_move=True,
    no_close=True,
    no_collapse=True,
    width=300
) as check_w:
    dpg.bind_item_theme(check_w, check_window)
    with dpg.table(header_row=False, width=100):
        for _ in range(3):
            dpg.add_table_column()
        for row in range(5):
            with dpg.table_row():
                for col in range(3):
                    cell = dpg.add_selectable(label="X", callback=check_table_callback, user_data=col * 5 + row)
                    dpg.bind_item_theme(cell, check_cell)
    with dpg.table(header_row=False, width=120):
        for _ in range(2):
            dpg.add_table_column()
        with dpg.table_row():
                b1 = dpg.add_button(label="SYNC", width=50, user_data=True, callback=check_result)
                b2 = dpg.add_button(label="ASYNC", width=50, user_data=False, callback=check_result)
                dpg.bind_item_theme(b1, check_buttons)
                dpg.bind_item_theme(b2, check_buttons)
    dpg.add_input_text(hint="RESULT", enabled=False, width=115, tag="result")

dpg.create_viewport(
    title="LAB7 by @therealmal",
    width=300,
    height=315,
    resizable=False)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()