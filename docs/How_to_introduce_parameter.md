# For code-developers: how to introduce new attributes in Quket

All the parameters should be defined as an attribute of `QuketData` class or its subclasses (`Config`, `Multi`, etc.), except for very essential, fixed ones such as bohr/angstrom conversion factor, which should go to `quket/config.py`. This document explains how one gets to introduce a new such parameter `xxx` with the default integer value `5`, which can be changed by setting an appropriate option in an input file or `quket.create()` arguments.

1.	QuketData and its subclasses are a `dataclass` defined in [quket_data/quket_data.py](../quket/quket_data/quket_data.py). Each of them defines their attributes and default values. For example, QuketData defines the `method` attribute with the default value `vqe` as
```
method: str = "vqe"
```
in the beginning of 
```
class QuketData():
```
If one wants to put the attribute `xxx` to `QuketData`, i.e., to use `QuketData.xxx`, simply add a line to define the attribute and its default value 5,
```
xxx: int = 5
```
If `xxx` is supposed to be other type such as `str`, change the type accordingly. If one wants instead to put the attribute to `Config` subclass (used as `QuketData.cf`), do the above in the definition of `Config` class (after `class Config():` in [quket_data/quket_data.py](../quket/quket_data/quket_data.py)).
When a `QuketData` instance is created, now it contains the attribute `xxx` with integer `5`.

2.	The next step is to be able to read an input that specifies a user-defined value for `xxx`. To do this, one simply the top of modifies [fileio/read.py](../quket/fileio/read.py), which defines parameter dictionaries. There are four different types: `integers`, `floats`, `bools`, `strings`. Keys correspond to the option names (those in the input file or `quket.create()` arguments) and values correspond to the attribute names (e.g., `xxx`). In most cases, they can be the same. In this example, let us add new key and value as `“yyy”: “xxx”` in the dictionary `integers`.

3.	Now the `quket.create()` function should be able to read the parameter correctly,
```
Q = quket.create(yyy=10)
Q.xxx 
```
which should return `10`.



# Quketで新たなパラメータを導入したい場合 (日本語版)
基本的にパラメータはすべて（ただしbohr/angstrom変換係数など非常に基本的な物理定数などは[config.py](../quket/config.py)に置く） `QuketData`やそのサブクラスのアトリビュートに入れるべきである。ここではあるパラメータ`xxx`を新たに導入し、そのデフォルト値が整数`5`になるようにし、またインプットや`quket.create()`関数の引数のオプションで他の値に変更できるようにするためにどうすればよいかを説明する。

1.	`QuketData`クラスやサブクラス （`QuketData.Config`など)のアトリビュートは[quket_data/quket_data.py](../quket/quket_data/quket_data.py)で定義されている。これらは`dataclass`であり、単純にアトリビュート名とデフォルト値を挿入すれば良い。例えば`QuketData`クラスは`method`アトリビュートを持っており、そのデフォルト値は`vqe`（文字列）である。これは
```
class QuketData():
```
の直下で
```
method: str = "vqe"
```
として宣言されている。今、`QuketData.xxx`として新たにアトリビュートを定義したい場合、同様に
```
xxx: int = 5
```
とすればよい。上記の代わりに`QuketData.cf` (`Config`サブクラス) のアトリビュートとして定義したい場合は
```
Class Config():
```
を同じようにいじれば良い。

2.	次に、インプットや`quket.create()`の引数オプションを指定することでアトリビュートの値を変更できるようにする。[fileio/read.py](../quket/fileio/read.py)のトップ、
```
# How to write; 'option's name': 'attribute's name'
```
の下にパラメータ（アトリビュート）の辞書が定義されている。これらは便宜上、`integers`, `floats`, `bools`, `strings`の4つに別れている。キーKeyはオプションの名前 (インプット内でのオプション名もしくは`quket.create()`の引数名)であり、値Valueはアトリビュートの名前である。多くの場合、これらは同じものでよいが、今回は前者を`yyy`、後者を`xxx`とすることにする。この場合、`xxx`は整数タイプなので`integers`辞書に
```
“yyy”: “xxx”
```
を挿入する。

3.	こうして変更した`quket`モジュールをインポートしてやればアトリビュートとしてオプションを読み込ませることができる。
```
Q = quket.create(yyy=10)
Q.xxx 
```
を実行すれば10が得られる。

