import re
import cv2
from pathlib import Path


def tsne_video(root, save_path=None):
    image_files = list(root.rglob('*-tsne.png'))
    image_files.sort(key=lambda x: int(x.stem.split('-')[0]))

    vid_writer = None
    save_path = str((save_path / 'tsne').with_suffix('.mp4'))  # force *.mp4 suffix on results videos
    for image_file in image_files:
        img = cv2.imread(str(image_file))

        cv2.putText(img, f"Epoch on {int(image_file.stem.split('-')[0])}",
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0),
                    2, cv2.LINE_4)

        if vid_writer is None:
            fps, w, h = 5, img.shape[1], img.shape[0]
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(img)
    vid_writer.release()
