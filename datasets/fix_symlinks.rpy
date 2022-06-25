def start():
    #Call this function in /mnt/Noman/Ubuntu/CleanCode/Scrapbook/archived_experiments/Alphacube/datasets
    ans=get_all_paths(explore_symlinks=False,recursive=True,relative=False,physical=False,include_files=False)
    ans=[x for x in ans if is_symlink(x)]
    for x in ans:
        print(x,read_symlink(x))
        old_prefix='/mnt/Noman/Ubuntu/CleanCode/'
        new_prefix='/home/Anonymous/CleanCode/'
        a,b=x,read_symlink(x)
        if b.startswith(old_prefix):
            os.remove(a)
            b=new_prefix+b[len(old_prefix):]
            make_symlink(a,b)
            print('Symlinking    ',a,'   ---->   ',b)