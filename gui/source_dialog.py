from typing import Optional, Dict

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog

VIDEO_FILTER = 'Video files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)'


class SourceSelectionDialog(QDialog):
    """Lets the user pick a video file or a folder of frame images to label."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('ATLAS-Interactive')
        self.setMinimumWidth(420)
        self.result_source: Optional[Dict[str, str]] = None

        layout = QVBoxLayout(self)

        title = QLabel('What would you like to label?')
        title.setStyleSheet('font-size: 16px; font-weight: bold;')
        layout.addWidget(title)

        subtitle = QLabel('Choose a video file, or a folder that already contains frame images.')
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        video_button = QPushButton('🎥  Choose Video File...')
        video_button.setMinimumHeight(40)
        video_button.clicked.connect(self._choose_video)
        layout.addWidget(video_button)

        folder_button = QPushButton('📁  Choose Image Folder...')
        folder_button.setMinimumHeight(40)
        folder_button.clicked.connect(self._choose_folder)
        layout.addWidget(folder_button)

        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

    def _choose_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Choose Video File', '', VIDEO_FILTER)
        if file_name:
            self.result_source = {'video': file_name}
            self.accept()

    def _choose_folder(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Choose Image Folder', '')
        if dir_name:
            self.result_source = {'images': dir_name}
            self.accept()


def prompt_for_source() -> Optional[Dict[str, str]]:
    """Ask the user to pick a video file or an image folder.

    Returns {'video': path} or {'images': path}, or None if the user cancelled.
    """
    dialog = SourceSelectionDialog()
    dialog.exec()
    return dialog.result_source
