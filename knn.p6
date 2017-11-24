use v6;
use lib '.';
use CSV;

sub euclidian-distance(@p1, @p2) {
    (@p1 Z- @p2).map(* **2).sum.sqrt;
}

sub infix:«|<--->|»(@p1, @p2) {
    euclidian-distance(@p1, @p2);
}

sub prefix:<★★★★★>(@p1) {
    my %count;
    %count{$_}++ for @p1;
    my @maxes = %count.kv.flat.grep(-> $a, $b { $b == %count.values.max });
    return @maxes.elems == 1
        ?? @maxes[0][0]
        !! ★★★★★ @p1[0..^(@p1.elems-1)];
}

sub 死ね(*@a) {
    die q「お前はもう死んでいる: 」 ~ join(' ', @a);
}

sub 何(@a) {
    @a.say;@a
}

sub MAIN(Str :$known-file   = 'dataset1.csv',
         Str :$unknown-file = 'days.csv',
         Int :$k = 5) {

    死ね q「何！」 unless (^42).roll;

    my @known         = CSV::parse($known-file.IO.lines);
    my @known-seasons = map ((* % 12) div 3), map *[0].substr(4,2), @known;

    my $cdr = *[1..*];

    my @unknown = CSV::parse($unknown-file.IO.lines);

    #dieboeg:
    #.say for map(-> $x { map(* |<--->| $x, map($cdr, @known) Z @known-seasons) }, map($cdr, $unknown);

    my @unknown-seasons =
         (map($cdr, @unknown)
          .map(-> $x { ★★★★★ 何(((map(* |<--->| $x, map($cdr, @known)) Z @known-seasons)
                                 .sort(*[0] <=> *[0]))[0..^$k]
                                .map(*[1])) }));

    .say for map({ <winter lente zomer herfst>[$_] }, eager @unknown-seasons);
}
