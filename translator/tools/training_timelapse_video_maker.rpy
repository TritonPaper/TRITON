#EXAMPLE USAGE DIRECTORY:
# cd ~/CleanCode/Projects/TRITON/Versions/Alphacube__alphabet_five/translator/trained_models/outputs/alphabet_five_base__just_tex_reality__run6_branch1/images
 
rows=7      #How many rows are in these images?
columns=16  #How many columns are in these images?
row=4       #Which row do we want a timelapse of?
iter_per_image = 250 #How many iterations pass by every image?

def get_frame(file,row=4,rows=7,columns=16):
    image=load_image(file)
    chunks=split_tensor_into_regions(image,rows,columns,flat=False)
    images=chunks[row]
    return tiled_images(images)

def start():
    name=get_folder_name(get_parent_directory(get_current_directory()))
    print('Name:',name)

    title=''
    title+='training_timelapse___'
    title+=name
    title+='.mp4'

    files=get_all_files(relative=True)
    files=[x for x in files if x.startswith('gen_a2b_test_')]
    files.sort()

    print('Number of frames:',len(files))
    video_path=path_join('../../../../../untracked',title)
    video_writer=VideoWriterMP4(video_path,video_bitrate='max',framerate=20)
    try:
        for i,file in enumerate(files):
            frame=get_frame(file)
            #Iteration label is approximate, give or take a small few
            frame=labeled_image(frame,'%s  -  Iter %6i'%(name,iter_per_image*i))
            video_writer.write_frame(frame)
            #print(i)
    finally:
        video_writer.finish()
        print('Done!')
        print('Created',video_path)
        print('AKA',get_absolute_path(video_path))
