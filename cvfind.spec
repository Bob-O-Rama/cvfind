#
# spec file for package cvfind
#
Name:           cvfind
Version:        0.1.0
Release:        1%{?dist}
Summary:        An alternative to Hugin cpfind and panotools project cleanup utility

License:        Apache-2.0
URL:            https://github.com/Bob-O-Rama/cvfind
Source:         cvfind.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-build

BuildRequires:  opencv
BuildRequires:  opencv-devel
BuildRequires:  cmake
BuildRequires:  gcc > 12
BuildRequires:  gcc-c++ > 12
BuildRequires:  zlib-devel
BuildRequires:  zlib
BuildRequires:  unzip
# BuildRequires:  pkgconfig(opencv4)

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
echo '%prep invoked'
# %setup -q -n %{name}-%{version}

%setup -c %{name}-%{version}
echo '%setup -c ' %{name}-%{version} invoked

%configure
echo '%configure invoked'

%build
echo '%build invoked'
echo '- - - - -'
echo "`cat Makefile`"
echo '- - - - -'
echo "`pkg-config --debug opencv4`"
echo '- - - - -'
echo "g++ -L`pkg-config --variable=libdir opencv4` `pkg-config --cflags-only-I opencv4` -Wl,-rpath=/usr/local/lib64/ `pkg-config --libs opencv4` -g3 cvfind.cpp -o cvfind"
echo '- - - - -'
echo 'Invoking make...'
make check

# Apparently ignored
%check
echo '%check invoked'
make check

%install
echo '%install invoked'
rm -rf $RPM_BUILD_ROOT
make install PREFIX=/usr DESTDIR=$RPM_BUILD_ROOT 
# May be unnecessary, make install puts it in the right spot
# install -m 755 -d $RPM_BUILD_ROOT/%{_sbindir}
# ln -s cvfind $RPM_BUILD_ROOT/%{_sbindir}

# %find_lang %{name}
# %files -f %{name}.lang
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

