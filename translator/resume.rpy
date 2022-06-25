
try:
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b)
                trainer.gen_update(images_a, images_b)
                torch.cuda.synchronize()
            trainer.update_learning_rate()

            # Dump training stats in log file
            if (iterations + 1) % config.log_iter == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config.image_save_iter == 0:
                with torch.no_grad():
                    test_image_outputs  = trainer.sample(test_display_images_a , test_display_images_b )
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs  , display_size , image_directory , 'test_%08d'  % (iterations + 1 ))
                write_2images(train_image_outputs , display_size , image_directory , 'train_%08d' % (iterations + 1 ))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config.image_save_iter, 'images')
                del test_image_outputs, train_image_outputs

            if (iterations + 1) % config.image_display_iter == 0:
                with torch.no_grad():
                    test_image_outputs  = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_current')
                write_2images(train_image_outputs, display_size, image_directory, 'train_current')
                del test_image_outputs, train_image_outputs

            # Save network weights
            if (iterations + 1) % config.snapshot_save_iter == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')
except BaseException as exception:
    from rp import print_verbose_stack_trace
    print_verbose_stack_trace(exception)
