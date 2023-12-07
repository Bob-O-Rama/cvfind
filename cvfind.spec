Name: cvfind
​Version: 0.0.1
​Release: 1%{?dist}
​Summary: An alternative to Hugin cpfind and panotools project cleanup utility
​
​License: GPLv2+
​URL: https://github.com/Bob-O-Rama/cvfind
​Source0: https://github.com/Bob-O-Rama/cvfind/archive/refs/heads/main.zip
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)
​
BuildRequires: opencv
BuildRequires: opencv-devel
BuildRequires: cmake
BuildRequires: gcc 
BuildRequires: gcc-c++ 
BuildRequires: zlib-devel
BuildRequires: zlib

​%description
​cvfind is a robust alternative to Hugin's cpfind control point detector.
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
​
​%prep
​%setup -q -n
​
​%build
​%configure
​make %{?_smp_mflags}
​
​%check
​make check
​
​%install
​rm -rf $RPM_BUILD_ROOT
​make install DESTDIR=$RPM_BUILD_ROOT
​
​install -m 755 -d $RPM_BUILD_ROOT/%{_sbindir}
​ln -s ../bin/cvfind $RPM_BUILD_ROOT/%{_sbindir}
​
​%find_lang %{name}
​
​%files -f %{name}.lang
​%doc README.md
​%{_bindir}/*
​%{_sbindir}/*
​%{_mandir}/man1/*
​
​%changelog
​* Wed Dec 06 2023 Bob Mahar <bob@muhlenberg.edu> 0.0.1
​- Initial RPM release