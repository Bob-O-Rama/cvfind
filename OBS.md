# Open Build Service - Notes

Objective:

Enable cvfind to be automatically built, tested, and packaged by OBS for arbitrary *nix target platforms.

Status:

- Code cleanup to enable cvfind to be built against OpenCV when OpenCV is not built with "non-free" or opencv_contrib materials.
- Created OBS package shell to be used for checking in the rpmbuild, autobuild, obx materials needed for sandbox local builds
- Created script to reproduce an OBS local build environment in a repeatabale fashion.  Fetches OBS dependancies, makes local user accounts, fetches cvfind source from github, fetches OSB package shell, merges and prepares for local build.


