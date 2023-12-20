If you are looking to enrich your existing Python project with support for
formulae, you have come to the right place. Formulaic is designed with simple
APIs that should make it straightforward to integrate into any project.

In this document we provide several general recommendations for developers
integrating Formulaic, and then some more specific guidance for developers
looking to migrate existing formula functionality from `patsy`. As you are
working on integration Formulaic, if you come across anything not mentioned here
that really ought to be, please report it to our
[Issue Tracker](https://github.com/matthewwardrop/formulaic/issues).

## Recommendations

For the most part, Formulaic should "just work". However, here are a couple of
recommendations that might make your integration work easier.

* Do not use the user-facing syntactic sugar function `model_matrix`. This is a
  simple wrapper around lower-level APIs that automatically includes variables
  from users' local namespaces.  This is convenient when running in a notebook,
  but can lead to unexpected interactions with your library code that are hard
  to debug. **Called naively in your library it will treat the frame in which it
  was run as the user context, which may include somewhat sensitive internal
  state and may override transforms normally available to formulae.** Instead,
  use `Formula(...).get_model_matrix(...)`.
* If you do need access to user namespaces it is recommended that you use the
  `formulaic.utils.context.capture_context()` function and pass the result as
  `context` to the `.get_model_matrix()` methods. It is easiest to use in the
  outermost user-facing entrypoints so that you do not need to figure out
  exactly how many frames removed you are from user-context. You may also
  manually construct a dictionary from the user's context if you want to do
  additional filtering.
* During the evaluation of some term factors, the `eval()` function may be
  called to invoke the indicated Python functions. Since this is user-specified
  code, it is possible that the formula had some malicious code in it (such as
  `sys.exit()` or `shutil.rmtree()`). If you are integrating Formulaic into
  server-side code, it is highly recommended **not** to pass in any
  user-specified context, but instead to curate the set of additional functions
  that are available and pass that in instead. If you are writing a user-facing
  library, this should not be as concerning.
* Formulaic selects the materialization algorithms to use based on the incoming
  data type (e.g. `pandas.DataFrame` -> `PandasMaterializer`). Different
  materializers may have different output (and other) options. It may make sense
  to hard-code your choice of materializer by passing `materializer=` to the
  `.get_model_matrix()` methods.
* Formulaic typically provides sensible defaults that should work in most
  scenarios out-of-the-box. However, it may make sense for your library to
  override some of these defaults. For example, if you typically deal with
  categorical factors with high cardinality you may want to enable sparse
  outputs by default (by passing `output='sparse'` to `.get_model_matrix()`,
  assuming the materializer of your datatype supports this).
* Do not rely on `ModelSpec` instances to work across different major versions
  of Formulaic. It may be tempting to serialize them to disk and then reuse them
  in newer versions of Formulaic. Most of the time this will work fine, but the
  stored encoder and transform states are considered implementation details of
  stateful transforms and are subject to change between major versions. Patch
  releases should never result in changes to this state.

## Migrating from Patsy

If you are migrating a library that previous used `patsy` to `formulaic`, you
should first review the general user-facing [migration notes](../../migration/),
which describes differences in API and formula grammars. Then, in addition to
the recommendations above, the following notes might be helpful.

* While the vast majority of formulae will be parsed identically in Formulaic
  and Patsy, there will inevitably be small differences in some edge-cases (as
  highlighted in [migration notes](../../migration/)). For highly entrenched
  use-cases, do not expect this to be without friction.
* If you used any of the internals of `patsy`, such as manually assembling
  `Term` instances, this code will need to be rewritten to use `Formulaic`
  classes instead. Generally speaking, this will likely be transparent to your
  users and so should be a relatively small lift.
* Formulaic is much more flexible than Patsy, and so moulding it to your needs
  should be much easier. If you do need assistance, please always feel free to
  open an issue in our [issue tracker](https://github.com/matthewwardrop/formulaic/issues).
