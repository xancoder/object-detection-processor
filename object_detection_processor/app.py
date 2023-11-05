import datetime
import math
import pathlib

import cv2
import filetype
import pandas as pd
import streamlit as st
import ultralytics as ul


def main():
    # streamlit page
    st.set_page_config(
        page_title='Object Detection Processor',
        page_icon=':camera:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # sidebar - model options
    st.sidebar.header(
        'Object Detection Configuration'
    )

    model_input = st.sidebar.selectbox(
        label='Select Model',
        options=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        index=2,
        disabled=False
    )
    model_selected = f"models/{model_input}.pt"
    st.session_state['model'] = ul.YOLO(model_selected)

    classes_input = st.sidebar.multiselect(
        label='Object Classes',
        options=list(st.session_state['model'].names.values()),
        default=['person']
    )
    classes = []
    for c in classes_input:
        classes.append(get_number_of_class(st.session_state['model'].names, c))
    st.session_state['classes'] = classes

    confidence_input = st.sidebar.slider(
        label='Select Model Confidence',
        min_value=50,
        max_value=100,
        value=60
    )
    st.session_state['confidence'] = float(confidence_input) / 100

    device_input = st.sidebar.selectbox(
        label='Processing Device',
        options=['cpu', 'cuda - 0', 'cuda - 0,1'],
        index=0,
        disabled=False
    )
    if 'cuda' in device_input:
        st.session_state['device'] = device_input.split(' - ')[1]
    else:
        st.session_state['device'] = None

    # main content
    with (st.container()):
        src_input = st.text_input(
            label='Source'
        )
        if src_input == '':
            st.error('please enter a source')
            st.stop()
        if pathlib.Path(src_input).exists():
            st.toast('folder to process detected')
            init_processing_folder(src_input)
        else:
            st.error('invalid source input')


def init_processing_folder(src):
    col1, col2 = st.columns([.2, .8])
    with col1:
        recursive = st.toggle(
            label='include sub folder'
        )
        path_src = pathlib.Path(src)
        if recursive:
            files = path_src.glob('**/*')
        else:
            files = path_src.glob('*')
    with col2:
        if 'dst' not in st.session_state:
            st.session_state['dst'] = ''
            d = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')
            t = str(path_src.name) + f'_predict_{d}'
            t = path_src.parent / t
            st.session_state['dst'] = t
        dst_input = st.text_input(
            label='Destination Path',
            value=st.session_state['dst']
        )
        dst = pathlib.Path(dst_input)

    # processing
    selected_files = []
    with st.spinner(f'collecting data from folder: {path_src}'):
        for cfn in files:
            if cfn.is_dir():
                continue
            if 'predict' in str(cfn.absolute()):
                continue
            if filetype.is_video(cfn) or filetype.is_image(cfn):
                out_folder = dst / cfn.relative_to(src).parent / cfn.stem
                selected_files.append({
                    'Name': cfn.name,
                    'Path': cfn.absolute(),
                    'RelPath': cfn.relative_to(src).parent,
                    'Stem': cfn.stem,
                    'Size': convert_size(cfn.stat().st_size),
                    'DstFolder': out_folder
                })

    df = pd.DataFrame(selected_files)
    if df.empty:
        st.error('nothing found to process')
    else:
        st.success(f'found {df.shape[0]} items')
        run_btn = st.button('Start processing')
        if run_btn:
            if dst_input == '':
                st.error('enter destination folder')
                st.stop()
            progress_text = 'Operation in progress. Please wait.'
            progress_bar = st.progress(0, text=progress_text)
            with st.empty():
                for index, row in df.iterrows():
                    src = row['Path']
                    progress_text = f'Operation in progress. Please wait for: {int(index) + 1} / {df.shape[0]}'
                    percent_complete = int(((int(index) + 1) * 100) / df.shape[0])
                    progress_bar.progress(percent_complete, text=progress_text)
                    with st.spinner(f'Wait for processing: {src}'):
                        results = predict(
                            model=st.session_state['model'],
                            source=src,
                            threshold=st.session_state['confidence'],
                            device=st.session_state['device'],
                            classes=st.session_state['classes']
                        )
                        if filetype.is_video(src):
                            process_video(results, row['DstFolder'])
                        elif filetype.is_image(src):
                            process_image(results, row['DstFolder'])
                progress_bar.empty()
                st.write('processing finished')


def get_number_of_class(d: dict, search_item: str) -> str:
    for no, item in d.items():
        if item == search_item:
            return no


def convert_size(size_bytes):
    if size_bytes == 0:
        return '0 Bytes'
    size_name = ('Bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    return '%s %s' % (s, size_name[i])


def predict(model, source, threshold, device=None, classes=0):
    vid_stride = True  # False
    return model.predict(
        source=source,
        conf=threshold,
        imgsz=640,
        device=device,
        # save=True,
        classes=classes,
        stream=True,  # reduce potential out-of-memory
        vid_stride=vid_stride
    )


def process_image(results, of):
    for idx, r in enumerate(results):
        if not r.boxes:
            continue
        output_path = pathlib.Path(of)
        output_path.mkdir(parents=True, exist_ok=True)
        annotated_frame = r.plot()
        save_detected_image(annotated_frame, pathlib.Path(r.path), idx, output_path)


def process_video(results, of):
    output_path = pathlib.Path(of)
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, r in enumerate(results):
        boxes = r.boxes  # Boxes object for bbox outputs
        if not boxes:
            continue
        annotated_frame = r.plot()
        if (idx % 15) == 1:
            save_detected_image(annotated_frame, pathlib.Path(r.path), idx, output_path)


def save_detected_image(annotated_frame, current_file_path, idx, output_path):
    f = current_file_path.stem + f'_{idx:06d}_predict.jpg'
    o = output_path / f
    cv2.imwrite(str(o), annotated_frame)


if __name__ == '__main__':
    main()
