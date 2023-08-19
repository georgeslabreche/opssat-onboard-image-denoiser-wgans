class DirUtils:

    BASE_PATH = 'data/denoiser_training_data_split'

    def __init__(self, noise_type, noise_factor):
        self.noise_type = noise_type 
        self.noise_factor = noise_factor

    def get_train_clean_path(self):
        return f'{self.BASE_PATH}/training/unnoised/'

    def get_train_noise_path(self):
        return  f'{self.BASE_PATH}/training/noised/{self.noise_type}/{self.noise_factor}/'

    def get_val_clean_path(self):
        return f'{self.BASE_PATH}/validation/unnoised/'

    def get_val_noise_path(self):
        return f'{self.BASE_PATH}/validation/noised/{self.noise_type}/{self.noise_factor}/'

    def get_test_clean_path(self):
        return f'{self.BASE_PATH}/test/unnoised/'

    def get_test_noise_path(self):
        return f'{self.BASE_PATH}/test/noised/{self.noise_type}/{self.noise_factor}/'

    def get_checkpoint_path(self, filename=None):
        return f'data/checkpoint/{self.noise_type}/{self.noise_factor}/{filename if filename is not None else ""}'

    def get_gen_img_path(self, filename=None):
        return f'data/output/WGAN/fake/{self.noise_type}/{self.noise_factor}/{filename if filename is not None else ""}'
