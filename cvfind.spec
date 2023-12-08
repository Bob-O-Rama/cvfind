#
# spec file for package cvfind
#
Name:           cvfind
Version:        0.1.0
Release:        1%{?dist}
Summary:        An alternative to Hugin cpfind and panotools project cleanup utility

License:        Apache-2.0
URL:            https://github.com/Bob-O-Rama/cvfind
Source:         cvfind.zip
BuildRoot:      %{_tmppath}/%{name}-%{version}-build

BuildRequires:  opencv
BuildRequires:  opencv-devel
BuildRequires:  cmake
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  zlib-devel
BuildRequires:  zlib

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
%setup -q -n %{name}-%{version}

%build
%configure
make %{?_smp_mflags}

%check
make check

%install
rm -rf $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT

install -m 755 -d $RPM_BUILD_ROOT/%{_sbindir}
ln -s ../bin/cvfind $RPM_BUILD_ROOT/%{_sbindir}

%find_lang %{name}

%files -f %{name}.lang
%doc README.md USAGE.md
%{_bindir}/*
%{_sbindir}/*
%{_mandir}/man1/*

%changelog
* Thu Dec  7 2023 Robert Mahar <bob@muhlenberg.edu>
  First efforts to get cvfind building under OBS.

