# Instrcutions
- shape of the data should follow this rule : [batch,channels(sensors),timesteps,features[optional]]
- we want to be able to use different datasets and also use covariants
- we only want to focus on past time steps to train the ssl model (no t+1)
- we use the lejepa (/home/crispy/lejepa/2406.04853v2.pdf) method for self supervised learning:
    - create views of different data [batch, views,:]
    - model is an encoder + projector, the goal is to train the encoder to extract meaningful features and regularize via Sigreg, we can use a probe head to determine performance or down stream task

    inv_loss = (v_proj.mean(dim=1)-v_proj).square().mean() # aims to pull similar views together
    sigreg = self.sigreg(proj) # looks that the data in a batch follows isotropic gaussian, therefore the batch must be shuffeled, however I am not sure
- goal is to provide a framework for ssl for virtual sensing or forecasting etc.
- most critical is the data augmentation, we have to build a flexible approach that is scaleable
- we want to be able to easily exchange different encoders (LSTM, 1DConvMamba,Transformer,Spatio-TemporalGNN,...)