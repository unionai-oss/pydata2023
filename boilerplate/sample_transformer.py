class MyTypeTransformer(TypeTransformer[MyType]):
    def __init__(self):
        super().__init__(MyType)

    def get_literal_type(self, t) -> LiteralType:
        ...

    def to_literal(
        self,
        ctx: FlyteContext,
        python_val: MyType,
        python_type: Type[MyType],
        expected: LiteralType,
    ) -> Literal:
        ...

    def to_python_value(
        self, ctx: FlyteContext, lv: Literal, expected_python_type: Type[MyType]
    ) -> Optional[MyType]:
        ...

    def to_html(
        self, ctx: FlyteContext, python_val: MyType, expected_python_type: Type[MyType]
    ) -> str:
        ...
