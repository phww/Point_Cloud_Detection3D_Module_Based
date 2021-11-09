#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/24 上午10:39
# @Author : PH
# @Version：V 0.1
# @File : registry.py
# @desc :
class Registry:

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dick = dict()
        self._children = dict()

        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dick)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str
