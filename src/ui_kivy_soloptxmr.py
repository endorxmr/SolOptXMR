from json.tool import main
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty, StringProperty, BooleanProperty, ListProperty
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.layout import Layout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

import json


def json_ui_loader(filename):
    return json.load(open(f"ui-menus/{filename}.json", "r"))


class Login_form(GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_id(self):
        for id, widget in self.parent.ids.items():
            if widget.__self__ == self:
                return id

    def login(self):
        id = self.get_id()  # Sometimes the parent does not have any ids???
        user = self.ids.username.text
        pwd = self.ids.password.text
        # print(f"Button has been presed in form {id}. user: {user}, pass: {pwd}")
        print(f"Button has been presed in form. user: {user}, pass: {pwd}")


class MainMenu_Button(Button):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MainMenuLeft_Layout(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        main_json = json_ui_loader("main")
        for submenu in main_json["submenus"]:
            self.add_widget(MainMenu_Button(text=submenu))

    def load_screen(self, scr):
        id = self.get_id()  # Sometimes the parent does not have any ids???
        print(f"{id} - Trying to load {scr}")


# class MainMenuRight_Layout(BoxLayout):
# class ShowcaseScreen(Screen):
class MainMenuRight_Screen(Screen):
    fullscreen = BooleanProperty(False)

    def add_widget(self, *args, **kwargs):
        if 'content' in self.ids:
            return self.ids.content.add_widget(*args, **kwargs)
        return super(MainMenuRight_Screen, self).add_widget(*args, **kwargs)


class Miner_Widget(Label):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Miner_Layout(GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        miners = json_ui_loader("miners")
        for miner in miners["miners"]:
            description = f'{miner["name"]}\n{miner["hashrate"]} H/s\n{miner["watt_cur"]} W'
            self.add_widget(Miner_Widget(text=description))


# class MainMenu_GridLayout(BoxLayout):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)


class myButton(Button):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def on_press(self):
        print(f"IMMA PRESSIN MAH BUTTON")
        return super().on_press()


class SolOptXMRApp(App):

    index = NumericProperty(-1)
    screen_names = ListProperty([])

    def build(self):
        right_screen = ScreenManager()
        # return Builder.load_file("soloptxmr.kv")
        self.screens = {}
        self.available_screens = sorted([ "Batteries", "Installation", "Geo", "Inputs", "Miners", "Habits"])
        self.screen_names = self.available_screens
        self.available_screens = [f"screens/{fn}.kv".format(fn).lower() for fn in self.available_screens]
        self.go_next_screen()
        # return 
        Builder.load_file("ui_kivy_soloptxmr.kv")
        return right_screen

    def go_next_screen(self):
        self.index = (self.index + 1) % len(self.available_screens)
        screen = self.load_screen(self.index)
        sm = self.root.ids.right_screen
        sm.switch_to(screen, direction='left')
        self.current_title = screen.name
        self.update_sourcecode()
    
    def go_screen(self, idx):
        self.index = idx
        self.root.ids.right_screen.switch_to(self.load_screen(idx), direction='left')
    
    def load_screen(self, index):
        if index in self.screens:
            return self.screens[index]
        screen = Builder.load_file(self.available_screens[index])
        self.screens[index] = screen
        return screen


if __name__ == "__main__":
    SolOptXMRApp().run()