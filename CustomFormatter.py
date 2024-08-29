import logging
from PyQt5 import QtGui


class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.ERROR: ('[%(asctime)s] %(levelname)s:%(message)s', 'yellow'),
        logging.DEBUG: ('[%(asctime)s] %(levelname)s:%(message)s', 'white'),
        logging.INFO: ('[%(asctime)s] %(levelname)s:%(message)s', 'white'),
        logging.WARNING: ('[%(asctime)s] %(levelname)s:%(message)s', 'yellow')
    }

    def __init__(self, *args, **kwargs):
        # `datefmt` 매개변수를 사용하여 시간 포맷을 초단위까지만 설정
        super(CustomFormatter, self).__init__(datefmt='%Y-%m-%d %H:%M:%S', *args, **kwargs)

    def format(self, record):
        """
        로깅 레코드를 지정된 형식으로 포맷합니다.
        로그 레벨에 따라 색상이 적용된 HTML 포맷을 사용합니다.
        :param record: 로깅 레코드
        :return: 포맷된 로그 문자열
        """
        last_fmt = self._style._fmt
        opt = CustomFormatter.FORMATS.get(record.levelno)
        if opt:
            fmt, color = opt
            self._style._fmt = "<font color=\"{}\">{}</font>".format(QtGui.QColor(color).name(), fmt)
        res = logging.Formatter.format(self, record)
        self._style._fmt = last_fmt
        return res
