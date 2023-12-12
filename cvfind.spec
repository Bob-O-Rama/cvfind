#
# spec file for package cvfind - Rev. 20231212_1559
#
stamp=`date +%%Y%%m%%d`
Name:           cvfind
Version:        0.1.0
Release:        %{?stamp}
Summary:        An alternative to Hugin cpfind and panotools project cleanup utility

License:        Apache-2.0
URL:            https://github.com/Bob-O-Rama/cvfind
Source:         cvfind.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-build

BuildRequires:  opencv
BuildRequires:  opencv-devel
BuildRequires:  cmake
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  zlib-devel
BuildRequires:  zlib
BuildRequires:  unzip

%description
cvfind is a robust alternative to Hugin's cpfind control point detector.
It also functions as Hugin project cleanup tool to eliminate bad pairs
in existing panotools projects.   Specifically designed to address the
repeating patterns in microchip die photography, cvfind is also suitable
many tiling applications.  As a cpfind compatible wrapper for many
OpenCV feature detection and matching methods, it provide for more robust
control point detection for low contrast, blurry source images.

In addition, cvfind provides a highly effective homography based method to
eliminate bad pairs, properly register images with repeating patterns,
and automatically perform more computationally expensive recovery of
expected pairs.  cvfind also can be used from the command line to clean
up existing Hugin project and optionally output diagnostic images that
enable the rapid identification of overlap issues.

%prep

%setup -c %{name}-%{version}

%configure

%build
make check

# Apparently ignored?
%check
make check

%install
rm -rf $RPM_BUILD_ROOT
make install PREFIX=/usr DESTDIR=$RPM_BUILD_ROOT 

%files
%doc README.md USAGE.md
%license LICENSE
%{_bindir}/*
# %{_sbindir}/*
# Probabaly we need man pages... some day...
# %{_mandir}/man1/*

%changelog
* Thu Dec  7 2023 Robert Mahar <bob@muhlenberg.edu>
  First efforts to get cvfind building under OBS.

