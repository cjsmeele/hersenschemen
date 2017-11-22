use v6;
unit package CSV;

our sub parse(@lines) {
    return lazy gather {
        take(split /';'/, $_) for @lines;
    };
}
