from typing import Any

from vsview.api import WidgetPluginBase, hookimpl

from .main import NativeResPlugin


@hookimpl
def vsview_register_toolpanel() -> type[WidgetPluginBase[Any, Any]]:
    return NativeResPlugin


@hookimpl
def vsview_register_tooldock() -> type[WidgetPluginBase[Any, Any]]:
    return NativeResPlugin
