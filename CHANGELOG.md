# Changelog

## 1.4.0 (April 2023)

### Added 

- Update warnings mentioning semantic_segmentation step in pipeline. 
- Adapt tests to new check conf pandora's format. [#77]
- Add multiband classifications. [#76]

### Fixed

- Deletion of the pip install codecov of githubAction CI.

## 1.3.0 (March 2023)

### Added

- Multiband images compatibility. [#72]

### Changed

- Rename confidence name and adapt it to snakecase [#73]

### Fixed

- Test notebook from githubAction [#70]

## 1.2.0 (January 2023)

### Added

- Add 3SGM (Semantic Segmentation for SGM) method. 

### Changed

- Force python version >= 3.7. [#63]
- Delete MCCNN accurate penalties. [#67]
- Change values for default penalties of MC CNN fast [#65]
- Udpate dependency to Pandora 1.3.0 

## 1.1.1 (December 2021)

### Changed

Update python packaging.


## 1.1.0 (June 2021)

### Added

- Version handling with setuptools_scm
- Confidence can be used as weights for cost volume.
- Udpate dependency to Pandora 1.1.0 and libSGM 0.4.0

## 1.0.0 (March 2021)

### BREAKING CHANGE

- Penalties method : removes mc_cnn_penalty parameter, added mc_cnn_fast_penalty and mc_cnn_accurate_penalty parameters
  to be able to automatically load the penalties according to the type of measure. 
- Change compute_penalty to take images datasets as input.
- Update depency to Pandora 1.0.0 version.[#59]

## 0.6.0 (January 2021)

### Changed

- Move /conf to /tests directory [#49]
- Update depency to Pandora 0.5.0 version.

## 0.5.0 (December 2020)

### Changed

- Update dependency to libSGM 0.3.1 version (disable parallel python implementation of SGM)
- Update depency to Pandora 0.4.0 version.

### Fixed

- Penalties registration : fix the import of penalties [#46]

## 0.4.0 (November 2020)

### Added

### Changed

- Update dependency to Pandora 0.3.0 version.

### Fixed

-  min_cost_paths calculation: fix creation of temporary disparity map to avoid dangerous DataArray manipulation. [#43]

