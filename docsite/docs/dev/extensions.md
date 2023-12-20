Formulaic was designed to be extensible from day one, and nearly all of its
core functionality is implemented as "plugins"/"modules" that you can use as
examples for how extensions could be written. In this document we will provide
a basic high-level overview of the basic components of Formulaic that can
extended.

An important consideration is that while Formulaic offers extensible APIs, and
effort will be made not to break extension APIs without reason (and never in
patch releases), the safest place for you extensions is in Formulaic itself,
where they can be kept up to date and maintained (assuming the extension is not
overly bespoke). If you think your extensions might help others, feel free to
reach out via the [issue tracker](https://github.com/matthewwardrop/formulaic/issues)
and/or open a pull request.


## Transforms

Transforms are likely the most commonly extended feature of Formulaic, and also
likely the least valuable to upstream (since transforms are often domain
specific). Documentation for implementing transforms is described in detail in
the [Transforms](../../guides/transforms/) user guide.


## Materializers

Materializers are responsible for translating formulae into model matrices as
documented in the [How it works](../../guides/formulae/#materialization) user
guide. You need to implement a new materializer if you want to add support for
new input and/or output types.

Implementing a new materializer is as simple as subclassing the abstract class
`formulaic.materializers.FormulaMaterializer` (or one of its subclasses). This
base class defines the API expected by the rest of the Formulaic system. Example
implementations include
[pandas](https://github.com/matthewwardrop/formulaic/blob/main/formulaic/materializers/pandas.py)
and
[pyarrow](https://github.com/matthewwardrop/formulaic/blob/main/formulaic/materializers/arrow.py).

During subclassing, the new class is registered according to the various
`REGISTER_*` attributes if `REGISTER_NAME` is specified. This registration
allows looking up of the materializer by name through the `model_matrix()` and
`.get_model_matrix()` functions. You can always manually pass in your
materializer class explicitly without this registration.


## Parsers

Parsers translate a formula string to a set of terms and factors that are then
evaluated and assembled into the model matrix, as documented in the [How it
works](../../guides/formulae/#parsed-formulae) user guide. This is unlikely to
be necessary very often, but can be used to add additional formula operators,
or change the behavior of existing ones.

Formula parsers are expected to implement the API of
`formulaic.parser.types.FormulaParser`. The default implementation can be seen
[here](https://github.com/matthewwardrop/formulaic/blob/main/formulaic/parser/parser.py).
You can pass in custom parsers to `Formula()` via the `parser` and
`nested_parser` options (see inline documentation for more details).

If you are considering extending the parser, please do reach out via the
[issue tracker](https://github.com/matthewwardrop/formulaic/issues).
