# Globalization Subroutines

The following globalization subroutines are available.

```@index
Pages = ["globalization.md"]
```

## [Line Search Algorithms](@id line-search)

Line Searches have been moved to an external package. Take a look at the
[LineSearch.jl](https://github.com/SciML/LineSearch.jl) package and its
[documentation](https://sciml.github.io/LineSearch.jl/dev/).

## Radius Update Schemes for Trust Region

```@docs
RadiusUpdateSchemes
```

### Available Radius Update Schemes

```@docs
RadiusUpdateSchemes.Simple
RadiusUpdateSchemes.Hei
RadiusUpdateSchemes.Yuan
RadiusUpdateSchemes.Bastin
RadiusUpdateSchemes.Fan
RadiusUpdateSchemes.NLsolve
RadiusUpdateSchemes.NocedalWright
```
