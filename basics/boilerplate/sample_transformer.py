class MyTypeTransformer(TypeTransformer[MyType]):
    def __init__(self):
        super().__init__(MyType)

    def get_literal_type(self, t) -> LiteralType:
        ...

    def to_literal(self, ctx: FlyteContext, python_val: T, python_type: Type[T], expected: LiteralType) -> Literal:
        ...

    def to_python_value(self, ctx: FlyteContext, lv: Literal, expected_python_type: Type[T]) -> Optional[T]:
        ...

    def to_html(self, ctx: FlyteContext, python_val: T, expected_python_type: Type[T]) -> str:
        ...
