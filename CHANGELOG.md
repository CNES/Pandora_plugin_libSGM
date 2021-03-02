# Changelog

## Unreleased

### BREAKING CHANGE

- Penalties method : removes mc_cnn_penalty parameter, added mc_cnn_fast_penalty and mc_cnn_accurate_penalty parameters
  to be able to automatically load the penalties according to the type of measure. 

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


