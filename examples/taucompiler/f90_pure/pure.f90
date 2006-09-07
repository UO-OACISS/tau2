


pure function foo(x)

  integer, intent(in) :: x
  integer :: y
  foo = x + 5
end function foo


elemental function calculate(x)
  interface
     pure function foo(x)
       integer, intent(in) :: x
     end function foo
  end interface

  integer, intent(in) :: x
  integer :: y
  calculate = foo(x) + 5
end function calculate


program hello
  integer :: x

  x = 5
  x = calculate(x)
  print *, "hello"
  print *, x


end program hello
