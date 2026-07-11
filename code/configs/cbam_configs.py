emd_feature_paths = {
    'original': '../data/emd_features/original/PTBXL_Batch_Original_EMD_reduced_features.csv',
    'snr24': '../data/emd_features/mixed_snr24/mixed_snr24_MAT_Batch_EMD_reduced_features.csv',
    'snr12': '../data/emd_features/mixed_snr12/mixed_snr12_MAT_Batch_EMD_reduced_features.csv',
    'snr6': '../data/emd_features/mixed_snr6/mixed_snr6_DenoisedCSV_EMD_reduced_features.csv',
    'snr0': '../data/emd_features/mixed_snr0/mixed_snr0_DenoisedCSV_EMD_reduced_features.csv',
    'snrm6': '../data/emd_features/mixed_snrm6/mixed_snrm6_MAT_Batch_EMD_reduced_features.csv',
}


conf_cbam_xresnet1d101_late_fusion = {
    'modelname': 'cbam_xresnet1d101_late_fusion',
    'modeltype': 'cbam_xresnet1d_model',
    'parameters': dict(
        input_mode='late_fusion',
        use_cbam=True,
        fusion_type='concat',
        cbam_reduction=16,
        cbam_kernel_size=7,
        feature_hidden_dim=256,
        feature_embedding_dim=128,
        feature_dropout=0.3,
        fusion_hidden_dim=256,
        fusion_dropout=0.4,
        emd_feature_paths=emd_feature_paths,
        emd_scenario='original',
        waveform_scenario='original',
        missing_record_policy='drop',
        feature_log_transform=False,
        input_size=10.0,
        chunkify_train=False,
        chunkify_valid=False,
        epochs=50,
        lr=1e-2,
    )
}


conf_cbam_xresnet1d101_late_fusion_superdiagnostic = {
    'modelname': 'cbam_xresnet1d101_late_fusion_superdiagnostic',
    'modeltype': 'cbam_xresnet1d_model',
    'parameters': dict(conf_cbam_xresnet1d101_late_fusion['parameters'])
}


conf_cbam_xresnet1d101_ecg_only = {
    'modelname': 'cbam_xresnet1d101_ecg_only',
    'modeltype': 'cbam_xresnet1d_model',
    'parameters': dict(
        input_mode='ecg_only',
        use_cbam=True,
        fusion_type='concat',
        input_size=10.0,
        chunkify_train=False,
        chunkify_valid=False,
        epochs=50,
        lr=1e-2,
    )
}


conf_cbam_xresnet1d101_ecg_only_superdiagnostic = {
    'modelname': 'cbam_xresnet1d101_ecg_only_superdiagnostic',
    'modeltype': 'cbam_xresnet1d_model',
    'parameters': dict(conf_cbam_xresnet1d101_ecg_only['parameters'])
}


conf_cbam_xresnet1d101_feature_only = {
    'modelname': 'cbam_xresnet1d101_feature_only',
    'modeltype': 'cbam_xresnet1d_model',
    'parameters': dict(
        input_mode='feature_only',
        use_cbam=True,
        fusion_type='concat',
        cbam_reduction=16,
        cbam_kernel_size=7,
        feature_hidden_dim=256,
        feature_embedding_dim=128,
        feature_dropout=0.3,
        emd_feature_paths=emd_feature_paths,
        emd_scenario='original',
        waveform_scenario='original',
        missing_record_policy='drop',
        feature_log_transform=False,
        input_size=10.0,
        chunkify_train=False,
        chunkify_valid=False,
        epochs=50,
        lr=1e-2,
    )
}


conf_cbam_xresnet1d101_feature_only_superdiagnostic = {
    'modelname': 'cbam_xresnet1d101_feature_only_superdiagnostic',
    'modeltype': 'cbam_xresnet1d_model',
    'parameters': dict(conf_cbam_xresnet1d101_feature_only['parameters'])
}
