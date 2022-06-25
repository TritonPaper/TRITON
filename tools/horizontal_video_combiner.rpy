video_paths=['/Users/Anonymous/Downloads/animation(5).mp4', '/Users/Anonymous/Downloads/megavideo_320_alphbet_five_extracted.mp4', '/Users/Anonymous/Downloads/CycleganAnimation_320.mp4', '/Users/Anonymous/Downloads/cut_512_run1__no_teal!.mp4']
def start():
    videos=[load_video_stream(x) for x in video_paths]
    output_path='output.mp4'
    video_writer=VideoWriterMP4(output_path,video_bitrate='max')
    def max_dimensions(images):
        heights=[get_image_height(x) for x in images]
        widths=[get_image_width(x) for x in images]
        return max(heights),max(widths)
        
    for i,frames in enumerate(zip(*videos)):
        height,width=max_dimensions(frames)
        output_subframes=[cv_resize_image(x,(height,width)) for x in frames]
        output_frame=horizontally_concatenated_images(output_subframes)
        video_writer.write_frame(output_frame)
        print(i)
    video_writer.finish()