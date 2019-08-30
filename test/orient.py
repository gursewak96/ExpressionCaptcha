test_transpose(self):
        a = np.array([[1, 2], [3, 4]])
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        assert_raises(ValueError, lambda: a.transpose(0))
        assert_raises(ValueError, lambda: a.transpose(0, 0))
        assert_raises(ValueError, lambda: a.transpose(0, 1, 2))

    def test_sort(self):
        # test ordering for floats and complex containing nans. It is only
        # necessary to check the less-than comparison, so sorts that
        # only follow the insertion sort path are sufficient. We only
        # test doubles and complex doubles as the logic is the same.

        # check doubles
        msg = "Test real sort order with nans"
        a = np.array([np.nan, 1, 0])
        b = np.sort(a)
        assert_equal(b, a[::-1], msg)
        # check complex
        msg = "Test complex sort order with nans"
        a = np.zeros(9, dtype=np.complex128)
        a.real += [np.nan, np.nan, np.nan, 1, 0, 1, 1, 0, 0]
        a.imag += [np.nan, 1, 0, np.nan, np.nan, 1, 0, 1, 0]
        b = np.sort(a)
        assert_equal(b, a[::-1], msg)

        # all c scalar sorts use the same code with different types
        # so it suffices to run a quick check with one type. The number
        # of sorted items must be greater than ~50 to check the actual
        # algorithm because quick and merge sort fall over to insertion
        # sort for small arrays.
        a = np.arange(101)
        b = a[::-1].copy()
        for kind in ['q', 'm', 'h']:
            msg = "scalar sort, kind=%s" % kind
            c = a.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)

        # test complex sorts. These use the same code as the scalars
        # but the compare function differs.
        ai = a*1j + 1
        bi = b*1j + 1
        for kind in ['q', 'm', 'h']:
            msg = "complex sort, real part == 1, kind=%s" % kind
            c = ai.copy()
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
            c = bi.copy()
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
        ai = a + 1j
        bi = b + 1j
        for kind in ['q', 'm', 'h']:
            msg = "complex sort, imag part == 1, kind=%s" % kind
            c = ai.copy()
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
            c = bi.copy()
            c.sort(kind=kind)