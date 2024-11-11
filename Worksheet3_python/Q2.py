fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
fruits_with_only_two_vowels = [fruit for fruit in fruits if sum(fruit.count(vowel) for vowel in 'aeiou') == 2]
print(fruits_with_only_two_vowels)