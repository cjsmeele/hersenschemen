#!/usr/bin/env perl

# ga.pl - Convert iris (or other data) to nn:: vector code.
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

# GA functions {{{

# fitness wordt in biologie aangeduit als ω dus vandaar
sub omegafy {
    my $genotype = shift;
    my ($hl, $npl) = ($genotype->{hidden_layers}, $genotype->{neurons_per_layer});
    say $hl; say $npl;
    
}

# }}}

die "usage: $0\n"
  . "       population\n"
  . "       hidden-layers-min\n"
  . "       hidden-layers-max\n"
  . "       neurons-per-layer-min\n"
  . "       neurons-per-layer-max\n"
  . "       training-sessions\n"
    unless @ARGV == 6;

my ($population,
    $hidden_layers_min,
    $hidden_layers_max,
    $neurons_per_layer_min,
    $neurons_per_layer_max,
    $training_sessions) = map int(shift), 1..6;

my @children = map { hidden_layers     => $hidden_layers_min     + int(rand $hidden_layers_max     + 1 - $hidden_layers_min),
                     neurons_per_layer => $neurons_per_layer_min + int(rand $neurons_per_layer_max + 1 - $neurons_per_layer_min), }, 1..$population;

my @preferencial_genes = map omegafy($_), @children;

use Data::Dump;
dd@children; exit;

print <<EOF;
EOF
