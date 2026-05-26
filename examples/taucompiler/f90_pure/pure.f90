module math_ops
  implicit none

contains

  pure integer function foo(x)
    integer, intent(in) :: x
    foo = x + 5
  end function foo

  elemental integer function calculate(x)
    integer, intent(in) :: x
    calculate = foo(x) + 5
  end function calculate

end module math_ops


program hello
  use math_ops
  implicit none
  integer :: x

  x = 5
  x = calculate(x)
  print *, "hello"
  print *, x
end program hello
