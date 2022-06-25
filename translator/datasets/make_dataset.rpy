from rp import *

def make_dataset(name,fake_dir,real_dir):
    assert directory_exists(fake_dir)
    assert directory_exists(real_dir)
    
    #fake_dir=get_absolute_path(fake_dir)
    #real_dir=get_absolute_path(real_dir)

    make_directory(name)
    set_current_directory(name)
    
    fake_dir=get_relative_path(fake_dir)
    real_dir=get_relative_path(real_dir)
    
    make_symlink('test_fake'  ,fake_dir)
    make_symlink('train_fake' ,fake_dir)
    make_symlink('test_real'  ,real_dir)
    make_symlink('train_real' ,real_dir)

    set_current_directory('..')
