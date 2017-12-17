#!/usr/bin/env perl

# data-to-net-input.pl - Convert iris (or other data) to nn:: vector code.
# Copyright (C) 2017, Chris Smeele and Jan Halsema.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

use 5.12.0;
use warnings;

# Functionele kak omdat perl5=oud :( {{{

sub reduce(&@) {
    my ($c, $v, @l) = @_;
    $v = $c->($v, $_) for @l;
    $v
}
sub sum { reduce { (shift) + (shift) } 0, @_ }

sub min { reduce { my ($c, $x) = @_;
                   defined $c ? [$c,$x]->[$c>$x] : $x } undef, @_ }

sub max { reduce { my ($c, $x) = @_;
                   defined $c ? [$c,$x]->[$c<$x] : $x } undef, @_ }
sub zip {
    my @r;
    for my $i (0..min(map $#$_, @_))
        { push @r, [map $_[$_]->[$i], 0..$#_] }
    @r
}

# Mistakes were made.
sub fchomp { local $_ = shift; chomp; $_ }

# }}}

sub distance {
    sqrt sum map $_**2, map $_->[0] - $_->[1], zip shift, shift
}
sub nmeans {
    # Wait a sec, are we reimplementing stuff from kmeans.scm in Perl5?
    my @mean;
    for my $v (@_) { $mean[$_] += $v->[$_] for 0..$#$v }
    map $_/@_, @mean
}

sub split_archetypes {
    my ($n, @points) = @_;
    my @mean = nmeans(@points);
    my @sorted = sort { distance($a, \@mean) <=> distance($b, \@mean) } @points;
    [@sorted[0..$n-1]], [@sorted[$n..$#sorted]]
}

die "usage: $0 training-samples-per-label hidden-layers neurons-per-layer training-sessions [dataset]\n"
    unless @ARGV >= 4 and @ARGV <= 5;

my ($training_samples_per_label,
    $hidden_layers,
    $neurons_per_layer,
    $training_sessions) = map int(shift), 1..4;

# This does the thing.
my %labels; push @{$labels{$_->[$#$_]}}, [map 0+$_, @$_[0..$#$_-1]] for map [split /,/, fchomp $_], grep /\S/, <>;
my @labels = sort keys %labels;

say STDERR sprintf "Distribution of dataset: (%s)", join("|", map scalar(@{$labels{$_}}), @labels);

{
    my $smallest_label_size = min(map scalar(@{$labels{$_}}), @labels);
    die sprintf "error: samples per label too high (%d >= %d)\n",
                $training_samples_per_label,
                $smallest_label_size
        if $training_samples_per_label >= $smallest_label_size;
}

my %labels_split = map {
    $_ => [split_archetypes $training_samples_per_label, @{$labels{$_}} ]
} @labels;

# Archetypes into training set, rest goes to test set.

my @Atraining = map @{$labels_split{$_}->[0]}, @labels; # Gebruik Jan Modaal voor training.
my @Atest     = map @{$labels_split{$_}->[1]}, @labels; # Gebruik Chris Modaal voor testing.

my @Ytraining = map((([(0)x$_, 1, (0)x($#labels-$_)]) x @{$labels_split{$labels[$_]}->[0]}), 0..$#labels);
my @Ytest     = map((([(0)x$_, 1, (0)x($#labels-$_)]) x @{$labels_split{$labels[$_]}->[1]}), 0..$#labels);

say STDERR sprintf "Training set size:       %d", scalar @Atraining;
say STDERR sprintf "Test set size:           %d", scalar @Atest;

sub matrixify {
    my $name = shift;
    my $r;
    $r .= sprintf "Matrixd<%d,%d> $name {\n", scalar(@_), scalar(@{$_[0]});
    $r .= sprintf "    %s,\n", join(", ", @$_) for @_;
    $r .= "};\n";
    $r;
}

say "void run() {";

say matrixify("Ytrain", @Ytraining) =~ s/^/    /grm;
say matrixify("Atrain", @Atraining) =~ s/^/    /grm;
say matrixify("Ytest",  @Ytest)     =~ s/^/    /grm;
say matrixify("Atest",  @Atest)     =~ s/^/    /grm;

say sprintf "    auto net = nn::make_net<double,%d,%d,%d,%d>{};",
            scalar(@{$Atraining[0]}),
            scalar(@{$Ytraining[0]}),
            $hidden_layers,
            $neurons_per_layer;

print <<EOF;

    // Randomize initial weights.
    std::apply([](auto& ...x) {
                   (x.mip([](auto) {
                       return (double)rand()/RAND_MAX*2 - 1;
                   }) , ...);
               }, net);

    // Train net.
    for (int _ = 0; _ < $training_sessions; ++_)
        std::apply([&](auto&...x) { nn::train(Atrain, Ytrain, x...); }, net);

    // Run test set.
    auto A = std::apply([&](auto&...x) { return nn::forwards(Atest, x...); }, net);
    std::cout << A;

    // Assess test set results.
    int correct = 0;
    int total   = 0;
    for (uint r = 1; r <= A.nrows; ++r) {
        ++total;
        bool waarom_heeft_cpp_geen_continues_naar_outer_loops = false;
        for (uint c = 1; c <= A.ncols; ++c) {
            if ((A(r,c) > 0.5) != (Ytest(r,c) > 0.5)) {
                waarom_heeft_cpp_geen_continues_naar_outer_loops = true;
                break;
            }
        }
        if (!waarom_heeft_cpp_geen_continues_naar_outer_loops)
            ++correct;
    }
    std::cout << "MSE(all): " << nn::get_mse(A, Ytest) << "\\n";
    std::cout << "Correct:  " << correct << "/" << total << "\\n";
}
EOF
