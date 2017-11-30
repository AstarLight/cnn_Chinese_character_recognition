# -*- coding: utf-8 -*-

from deep_ocr.utils import trim_string

# I take characters from http://hanyu.iciba.com/zt/3500.html
data = u'''
壹 贰 叁 肆 伍 陆 柒 捌 玖 零 拾 佰 仟 万 亿 圆
'''

data = trim_string(data)

