# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:33:58 2020

@author: ZLT
"""

from PIL import Image, ImageFont

from handright import Template, handwrite

text = "我能吞下玻璃而不伤身体。"
template = Template(
    background=Image.new(mode="1", size=(1024, 2048), color=1),
    font_size=100,
    font=ImageFont.truetype(r"C:\Users\ZLT\AppData\Local\Microsoft\Windows\Fonts\李国夫手写体 常规.ttf"),
)
images = handwrite(text, template)
for im in images:
    assert isinstance(im, Image.Image)
    im.show()