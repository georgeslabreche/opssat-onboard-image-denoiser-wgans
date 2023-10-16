from train import train

if __name__ == '__main__':

    # train WGAN models for all the noise types and noise factors
#    for noise_type in ['fnp', 'cfnp']:
#        for noise_factor in [50, 100, 150, 200]:

    for noise_factor in [50, 100, 150, 200]:
        for noise_type in ['fnp', 'cfnp']:
            train(40, noise_type, noise_factor)