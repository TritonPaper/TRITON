video_path='/Users/Anonymous/Downloads/megavideo__alphabet_five_base__just_tex_reality__run6__iter_60000.mp4'
def start():
    first_frame=next(load_video_stream(video_path))
    height,width=get_image_dimensions(first_frame)
    width/=2
    height=(height-15-15)/2
    height=int(height)
    width=int(width)
    
    def crop_frame(frame):
        return frame[-height:,:width]
    
    video=load_video_stream(video_path)
    
    video_writer=VideoWriterMP4('output.mp4',video_bitrate='max')
    
    for i,frame in enumerate(video):
        frame=crop_frame(frame)
        video_writer.write_frame(frame)
        print(i)
    print('Done!')
    
    video_writer.finish()
    