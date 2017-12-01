use v6;
unit package CSV;

our sub parse(@lines) {
    return eager gather {
        take(split /';'/, $_) for @lines;
    };
}
