# Python Bindings for [`esmini`](https://esmini.github.io/)

From the esmini documentation:

> esmini is a software tool to play OpenSCENARIO files. It's provided both as a stand
> alone application and as a shared library for linking with custom applications. In
> addition some tools have been developed to support design and analysis of traffic
> scenarios.

While the existing [`pyesmini`](https://github.com/ebadi/pyesmini) library exists, it
hasn't been updated in a while (~3 years as of the time of writing this document).
Moreover, it uses `ctypes`, which can be a finicky tool to write bindings with.

The current library uses [`cffi`](https://cffi.readthedocs.io/en/latest/overview.html),
which is a lot more reliable in generating bindings for C libraries.
