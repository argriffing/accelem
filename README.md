expectation maximization acceleration

Required dependencies:
 * [Python 2.7+](http://www.python.org/)
 * [pip](https://pip.readthedocs.org/) (installation)
 * [git](http://git-scm.com/) (installation)
 * [numpy](http://www.numpy.org/) (arrays)

Optional dependencies:
 * [nose](http://readthedocs.org/docs/nose/en/latest/) (testing)
   - `$ pip install --user git+https://github.com/nose-devs/nose`
 * [coverage](http://nedbatchelder.com/code/coverage/) (test coverage)
   - `$ apt-get install python-coverage`


User
----

Install:

    $ pip install --user git+https://github.com/argriffing/accelem

Test:

    $ python -c "import accelem; accelem.test()"

Uninstall:

    $ pip uninstall accelem


Developer
---------

Install:

    $ git clone git@github.com:argriffing/accelem.git

Test:

    $ python runtests.py

Coverage:

    $ rm -rf htmlcov/
    $ python-coverage run runtests.py
    $ python-coverage html
    $ chromium-browser htmlcov/index.html

Build docs locally:

    $ sh make-docs.sh
    $ chromium-browser /tmp/nxdocs/index.html

Subsequently update online docs:

    $ git checkout gh-pages
    $ cp /tmp/nxdocs/. ./ -R
    $ git add .
    $ git commit -am "update gh-pages"
    $ git push

